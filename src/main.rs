mod datastore;
mod graph;
mod local_embeddings;
mod openai;

use crate::datastore::{DataStoreClient, PendingTopic};
use crate::openai::{ChatClient, ChatInteraction, Interaction, SourceSummary};
use ::serde::Serialize;
use base64::{engine::general_purpose, Engine as _};
use chrono::{Duration, Local, Utc};
use dotenvy::dotenv;
use fern::Dispatch;
use google_oauth::AsyncClient;
use hex;
use hmac::{Hmac, Mac};
use libsqlite3_sys::sqlite3_auto_extension;
use log::LevelFilter;
use rand::distributions::{Alphanumeric, DistString};
use rocket::form::{Form, FromForm};
use rocket::fs::FileServer;
use rocket::http::{Header, Status};
use rocket::request::{FromRequest, Outcome};
use rocket::response::{content, status, Responder};
use rocket::serde::{json::Json, Deserialize};
use rocket::{response, Request, Response, State};
use rocket_dyn_templates::Template;
use serde_json;
use sha2::Sha256;
use sqlite_vec::sqlite3_vec_init;
use std::env;
use std::fmt::Write as _;
use std::net::SocketAddr;
use std::path::PathBuf;
use thiserror::Error;

#[macro_use]
extern crate rocket;
#[macro_use]
extern crate log;

type HmacSha256 = Hmac<Sha256>;
const SESSION_TTL_HOURS: i64 = 24;

#[derive(Clone)]
struct AppConfig {
    session_secret: Vec<u8>,
    google_client_id: String,
    embeddings_url: String,
    llm_api_base: String,
    llm_model: String,
    llm_api_key: Option<String>,
    llm_max_tokens: u32,
    llm_temperature: f32,
    llm_top_p: f32,
    llm_top_k: u32,
    llm_enable_thinking: bool,
    data_dir: PathBuf,
}

impl AppConfig {
    fn from_env() -> Self {
        let speck_key_hex = env::var("JOTBOX_SPECK_KEY")
            .expect("JOTBOX_SPECK_KEY environment variable is required");
        let google_client_id = env::var("JOTBOX_GOOGLE_CLIENT_ID")
            .expect("JOTBOX_GOOGLE_CLIENT_ID environment variable is required");
        let embeddings_url = env::var("JOTBOX_EMBEDDINGS_URL")
            .expect("JOTBOX_EMBEDDINGS_URL environment variable is required");
        let llm_api_base =
            env::var("LLM_API_BASE").unwrap_or_else(|_| "http://jura:11434/v1".to_string());
        let llm_model = env::var("LLM_MODEL").unwrap_or_else(|_| "qwen3:8b".to_string());
        let llm_api_key = env::var("LLM_API_KEY")
            .or_else(|_| env::var("OPENAI_API_KEY"))
            .ok();
        let llm_max_tokens = env::var("LLM_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(8192);
        let llm_temperature = env::var("LLM_TEMPERATURE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.6);
        let llm_top_p = env::var("LLM_TOP_P")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.95);
        let llm_top_k = env::var("LLM_TOP_K")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(20);
        let llm_enable_thinking = env::var("LLM_ENABLE_THINKING")
            .ok()
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false);

        let session_secret = hex::decode(speck_key_hex.trim_start_matches("0x"))
            .expect("Unable to parse JOTBOX_SPECK_KEY as hex");
        assert!(
            session_secret.len() >= 16,
            "JOTBOX_SPECK_KEY must be at least 128 bits"
        );

        let data_dir = env::var("JOTBOX_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("data"));
        std::fs::create_dir_all(&data_dir).expect("Failed to create data directory");

        AppConfig {
            session_secret,
            google_client_id,
            embeddings_url,
            llm_api_base,
            llm_model,
            llm_api_key,
            llm_max_tokens,
            llm_temperature,
            llm_top_p,
            llm_top_k,
            llm_enable_thinking,
            data_dir,
        }
    }
}

#[post("/api/chat/generate", data = "<input>")]
async fn api_chat_generate(
    remote_addr: SocketAddr,
    _token: Token,
    input: Json<ChatInteraction>,
    config: &State<AppConfig>,
) -> status::Custom<content::RawJson<String>> {
    info!("{:?} -> /api/chat/generate", remote_addr);
    let chat_interaction = input.0;

    let chat_client = ChatClient::new(
        &config.llm_api_base,
        config.llm_api_key.clone(),
        &config.llm_model,
        config.llm_max_tokens,
        config.llm_temperature,
        config.llm_top_p,
        config.llm_top_k,
        config.llm_enable_thinking,
    );

    match chat_client
        .chat(&chat_interaction, &chat_interaction.context)
        .await
    {
        Ok(text) => {
            let response = ChatResponse {
                text,
                sources: chat_interaction.sources.clone(),
            };
            let json_response: String = serde_json::to_string_pretty(&response).unwrap();
            status::Custom(Status::Ok, content::RawJson(json_response))
        }
        Err(err) => {
            error!("Chat generation failed: {:?}", err);
            status::Custom(
                Status::ServiceUnavailable,
                content::RawJson(
                    r#"{"error":"Language model unavailable. Please try again."}"#.to_string(),
                ),
            )
        }
    }
}

// Database handled per-user via DataStoreClient

#[derive(Debug)]
struct TemplateWithHeader {
    data: Template,
    total_count_header: Header<'static>,
}

impl TemplateWithHeader {
    pub fn new(data: Template) -> Self {
        Self {
            total_count_header: Header::new(
                "Referrer-Policy",
                "no-referrer-when-downgrade".to_string(),
            ),
            data: data,
        }
    }
}

impl<'r> Responder<'r, 'r> for TemplateWithHeader {
    fn respond_to(self, request: &Request) -> response::Result<'r> {
        let mut build = Response::build();
        build.merge(self.data.respond_to(request)?);
        build.header(self.total_count_header);
        build.ok()
    }
}

struct Token(String);

#[derive(Debug, Error)]
enum ApiTokenError {
    #[error("invalid session token")]
    Invalid,
    #[error("session token expired")]
    Expired,
}
#[rocket::async_trait]
impl<'r> FromRequest<'r> for Token {
    type Error = ApiTokenError;

    async fn from_request(request: &'r Request<'_>) -> Outcome<Self, Self::Error> {
        let config = match request.rocket().state::<AppConfig>() {
            Some(cfg) => cfg,
            None => return Outcome::Error((Status::InternalServerError, ApiTokenError::Invalid)),
        };
        let token = request.headers().get_one("X-Session-Key");
        match token {
            Some(token) => {
                // check validity
                match collection_from_session(token, config) {
                    Ok(collection) => Outcome::Success(Token(collection)),
                    Err(_) => Outcome::Forward(Status::Unauthorized),
                }
            }
            None => Outcome::Forward(Status::Unauthorized),
        }
    }
}

// API endpoints

#[get("/api/graph")]
async fn api_graph(
    remote_addr: SocketAddr,
    token: Token,
    config: &State<AppConfig>,
) -> content::RawJson<String> {
    info!("{:?} -> /api/graph", remote_addr);
    let collection = token.0;
    let vs = DataStoreClient::new(&collection, &config.embeddings_url, &config.data_dir)
        .await
        .unwrap();
    let graph_result = vs.list_graph().await.unwrap();
    if !graph_result.pending_topics.is_empty() {
        let pending = graph_result.pending_topics.clone();
        spawn_topic_refinement_jobs(
            pending,
            config.inner().clone(),
            vs.collection_id(),
            vs.db_path(),
        );
    }
    let r: String = serde_json::to_string_pretty(&graph_result.graph).unwrap();
    content::RawJson(r)
}

#[get("/api/topics/pending")]
async fn api_topics_pending(
    remote_addr: SocketAddr,
    token: Token,
    config: &State<AppConfig>,
) -> content::RawJson<String> {
    info!("{:?} -> /api/topics/pending", remote_addr);
    let collection = token.0;
    let vs = DataStoreClient::new(&collection, &config.embeddings_url, &config.data_dir)
        .await
        .unwrap();
    let pending = vs.pending_topic_count().await.unwrap_or(0);
    let r = serde_json::json!({ "pending": pending }).to_string();
    content::RawJson(r)
}

#[get("/api/recent")]
async fn api_recent(
    remote_addr: SocketAddr,
    token: Token,
    config: &State<AppConfig>,
) -> content::RawJson<String> {
    info!("{:?} -> /api/recent", remote_addr);
    let collection = token.0;
    let vs = DataStoreClient::new(&collection, &config.embeddings_url, &config.data_dir)
        .await
        .unwrap();
    let docs = vs.list_recent(6).await.unwrap();
    let r: String = serde_json::to_string_pretty(&docs).unwrap();
    content::RawJson(r)
}

#[post("/api/related", data = "<input>")]
async fn api_related(
    remote_addr: SocketAddr,
    token: Token,
    input: String,
    config: &State<AppConfig>,
) -> content::RawJson<String> {
    info!("{:?} -> /api/related", remote_addr);
    let collection = token.0;
    let vs = DataStoreClient::new(&collection, &config.embeddings_url, &config.data_dir)
        .await
        .unwrap();
    let docs = vs.list_related(&input, 6).await.unwrap();
    let r: String = serde_json::to_string_pretty(&docs).unwrap();
    content::RawJson(r)
}

#[get("/api/retrieve/<id>")]
async fn api_retrieve(
    remote_addr: SocketAddr,
    token: Token,
    id: i64,
    config: &State<AppConfig>,
) -> content::RawJson<String> {
    info!("{:?} -> /api/retrieve/{}", remote_addr, id);
    let collection = token.0;
    let vs = DataStoreClient::new(&collection, &config.embeddings_url, &config.data_dir)
        .await
        .unwrap();
    let doc = vs.retrieve_doc(id).await.unwrap();
    let r: String = serde_json::to_string_pretty(&doc).unwrap();
    content::RawJson(r)
}

#[get("/api/delete/<id>")]
async fn api_delete(
    remote_addr: SocketAddr,
    token: Token,
    id: i64,
    config: &State<AppConfig>,
) -> content::RawJson<String> {
    info!("{:?} -> /api/delete/{}", remote_addr, id);
    let collection = token.0;
    let vs = DataStoreClient::new(&collection, &config.embeddings_url, &config.data_dir)
        .await
        .unwrap();
    let _status = vs.delete_doc(id).await.unwrap();
    content::RawJson("{}".to_string())
}

#[derive(Debug, Deserialize)]
pub struct SaveRequest {
    pub id: i64,
    pub ciphertext: String,
    pub plaintext: String,
    pub title: String,
}

#[derive(Debug, Deserialize)]
#[serde(crate = "rocket::serde")]
struct ChatRetrieveRequest {
    query: String,
    history: Vec<Interaction>,
}

#[derive(Debug, Serialize)]
struct ChatRetrieveDocument {
    id: i64,
    title: String,
    score: i32,
}

#[derive(Debug, Serialize)]
struct ChatRetrieveResponse {
    documents: Vec<ChatRetrieveDocument>,
}

#[derive(Serialize)]
struct ChatResponse {
    pub text: String,
    pub sources: Vec<SourceSummary>,
}

#[post("/api/save", data = "<input>")]
async fn api_save(
    remote_addr: SocketAddr,
    token: Token,
    input: Json<SaveRequest>,
    config: &State<AppConfig>,
) -> Result<content::RawJson<String>, status::Custom<content::RawJson<String>>> {
    info!("{:?} -> /api/save", remote_addr);

    let save_req = input.0;
    let collection = token.0;
    let vs = DataStoreClient::new(&collection, &config.embeddings_url, &config.data_dir)
        .await
        .unwrap();

    let doc = vs
        .save_document(
            if save_req.id > 0 {
                Some(save_req.id)
            } else {
                None
            },
            &save_req.ciphertext,
            &save_req.plaintext,
            &save_req.title,
        )
        .await
        .map_err(|err| {
            error!(
                "Failed to persist document for {:?}: {:?}",
                remote_addr, err
            );
            status::Custom(
                Status::ServiceUnavailable,
                content::RawJson(
                    r#"{"error":"Embedding service is unavailable. Please try again later."}"#
                        .to_string(),
                ),
            )
        })?;

    Ok(content::RawJson(format!("{{ \"id\": {}}}", doc.id)))
}

fn spawn_topic_refinement_jobs(
    pending: Vec<PendingTopic>,
    config: AppConfig,
    collection_id: i64,
    db_path: PathBuf,
) {
    for topic in pending {
        if topic.doc_titles.is_empty() {
            continue;
        }
        let cfg = config.clone();
        let topic_db = db_path.clone();
        rocket::tokio::spawn(async move {
            if let Err(err) = refine_topic_label(topic_db, cfg, collection_id, topic).await {
                error!("Topic title refinement failed: {:?}", err);
            }
        });
    }
}

async fn refine_topic_label(
    db_path: PathBuf,
    config: AppConfig,
    collection_id: i64,
    topic: PendingTopic,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if topic.doc_titles.is_empty() {
        return Ok(());
    }

    let prompt = build_topic_prompt(&topic.doc_titles);
    let chat_client = ChatClient::new(
        &config.llm_api_base,
        config.llm_api_key.clone(),
        &config.llm_model,
        config.llm_max_tokens,
        (config.llm_temperature + 0.2).min(0.9),
        config.llm_top_p,
        config.llm_top_k,
        config.llm_enable_thinking,
    );
    let interaction = ChatInteraction {
        history: vec![],
        query: prompt,
        context: "".to_string(),
        sources: vec![],
    };
    let response = chat_client.chat(&interaction, "").await?;
    let fallback = format!("Topic {}", topic.group_id + 1);
    let label = sanitize_topic_label(&response, &fallback);

    let mut conn = DataStoreClient::connect_to_path(&db_path).await?;

    sqlx::query(
        "UPDATE topics SET label = ?, status = 'final', doc_count = ? WHERE collection_id = ? AND group_id = ?",
    )
    .bind(&label)
    .bind(topic.doc_count as i64)
    .bind(collection_id)
    .bind(topic.group_id as i64)
    .execute(&mut conn)
    .await?;
    sqlx::query("UPDATE documents SET topic = ? WHERE collection_id = ? AND topic_group = ?")
        .bind(&label)
        .bind(collection_id)
        .bind(topic.group_id as i64)
        .execute(&mut conn)
        .await?;
    Ok(())
}

fn build_topic_prompt(titles: &[String]) -> String {
    let mut prompt = String::from(
        "You are labeling a cluster of personal knowledge base notes. \
Return a short (4 words or fewer) descriptive title without quotes.\n\nNote titles:\n",
    );
    for (idx, title) in titles.iter().enumerate() {
        let _ = writeln!(&mut prompt, "{}. {}", idx + 1, title);
    }
    prompt.push_str("\nRespond with only the title.");
    prompt
}

fn sanitize_topic_label(raw: &str, fallback: &str) -> String {
    let candidate = raw
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or(fallback)
        .trim();
    let cleaned = candidate
        .trim_matches(|c: char| c == '"' || c == '\'' || c == '#' || c == '*' || c == '`')
        .trim();
    let result: String = cleaned.chars().take(64).collect();
    if result.is_empty() {
        fallback.to_string()
    } else {
        result
    }
}

#[post("/api/chat/retrieve", data = "<input>")]
async fn api_chat_retrieve(
    remote_addr: SocketAddr,
    token: Token,
    input: Json<ChatRetrieveRequest>,
    config: &State<AppConfig>,
) -> status::Custom<content::RawJson<String>> {
    info!("{:?} -> /api/chat/retrieve", remote_addr);
    let payload = input.0;
    let mut query = String::new();
    for i in payload.history.iter().rev().take(2) {
        query.push_str(&i.message);
    }
    query.push_str(&payload.query);

    let collection = token.0;
    let vs = match DataStoreClient::new(&collection, &config.embeddings_url, &config.data_dir).await
    {
        Ok(client) => client,
        Err(err) => {
            error!("Failed to create datastore client: {:?}", err);
            return status::Custom(
                Status::InternalServerError,
                content::RawJson(r#"{"error":"Unable to access datastore"}"#.to_string()),
            );
        }
    };

    match vs.list_related(&query, 5).await {
        Ok(documents) => {
            let docs: Vec<ChatRetrieveDocument> = documents
                .into_iter()
                .map(|doc| ChatRetrieveDocument {
                    id: doc.id,
                    title: doc.title,
                    score: doc.score,
                })
                .collect();
            let body = serde_json::to_string(&ChatRetrieveResponse { documents: docs }).unwrap();
            status::Custom(Status::Ok, content::RawJson(body))
        }
        Err(err) => {
            error!("list_related failed: {:?}", err);
            status::Custom(
                Status::ServiceUnavailable,
                content::RawJson(
                    r#"{"error":"Embeddings search unavailable. Please try again."}"#.to_string(),
                ),
            )
        }
    }
}

#[derive(Serialize)]
struct IndexData {
    error: String,
    google_client_id: String,
}

#[derive(Serialize)]
struct NotesData {
    name: String,
    session_id: String,
    user_sub: String,
}

#[get("/")]
fn index(remote_addr: SocketAddr, config: &State<AppConfig>) -> TemplateWithHeader {
    let index_data = IndexData {
        error: "".to_string(),
        google_client_id: config.google_client_id.clone(),
    };

    info!("{:?} -> /", remote_addr);
    TemplateWithHeader::new(Template::render("index", &index_data))
}

#[derive(FromForm, Debug)]
struct TokenForm<'r> {
    token: &'r str,
}

#[post("/", data = "<token_form>")]
async fn notes(
    remote_addr: SocketAddr,
    token_form: Form<TokenForm<'_>>,
    config: &State<AppConfig>,
) -> TemplateWithHeader {
    info!("{:?} -> /", remote_addr);

    let client = AsyncClient::new(&config.google_client_id);
    let id_token = client.validate_id_token(token_form.token).await;
    match id_token {
        Ok(token) => {
            let name = match token.given_name {
                Some(n) => n,
                None => "Google User".to_string(),
            };
            let (collection_name, session_id) = create_session_id(&token.sub, config.inner());
            info!(
                "{:?} -> / Sign-in: {}, {}, [{}]",
                remote_addr, &name, &collection_name, &session_id
            );

            // Create an embeddings table for this user
            let ds =
                DataStoreClient::new(&collection_name, &config.embeddings_url, &config.data_dir)
                    .await
                    .unwrap();
            let _ = ds.check_embeddings().await;

            let notes_data = NotesData {
                name,
                session_id,
                user_sub: token.sub.clone(),
            };
            TemplateWithHeader::new(Template::render("notes", &notes_data))
        }
        Err(e) => {
            error!("Error validating Google token: {:?}", e);
            info!("{:?} -> / Sign-in failed.", remote_addr);

            let index_data = IndexData {
                error: format!("Google sign-in failed: {:?}", e),
                google_client_id: config.google_client_id.clone(),
            };
            TemplateWithHeader::new(Template::render("index", &index_data))
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    dotenv().ok();
    unsafe {
        sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
    }

    // Initialise the logger
    Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{} [{}] [{}] {}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                record.target(),
                message
            ))
        })
        .level(LevelFilter::Info)
        .chain(std::io::stdout())
        //.chain(fern::log_file("jarvis.log").unwrap())
        .apply()
        .expect("Can't initialise logging");

    // Configure rocket
    let app_config = AppConfig::from_env();

    let _ = rocket::build()
        .attach(Template::fairing())
        .mount(
            "/",
            routes![
                index,
                notes,
                api_recent,
                api_related,
                api_retrieve,
                api_delete,
                api_chat_retrieve,
                api_chat_generate,
                api_save,
                api_graph,
                api_topics_pending
            ],
        )
        .mount("/", FileServer::from("html"))
        .manage(app_config)
        .launch()
        .await;
}

#[derive(::serde::Serialize, ::serde::Deserialize)]
struct SessionPayload {
    collection: String,
    exp: i64,
    nonce: String,
}

fn sign_payload(config: &AppConfig, payload: &[u8]) -> Vec<u8> {
    let mut mac =
        HmacSha256::new_from_slice(&config.session_secret).expect("Invalid session secret");
    mac.update(payload);
    mac.finalize().into_bytes().to_vec()
}

fn create_session_id(sub: &str, config: &AppConfig) -> (String, String) {
    let collection_name = format!("G{}Z", sub);
    let payload = SessionPayload {
        collection: collection_name.clone(),
        exp: (Utc::now() + Duration::hours(SESSION_TTL_HOURS)).timestamp(),
        nonce: Alphanumeric.sample_string(&mut rand::thread_rng(), 16),
    };

    let payload_json =
        serde_json::to_string(&payload).expect("Failed to serialize session payload");
    let signature = sign_payload(config, payload_json.as_bytes());
    let token = format!(
        "{}.{}",
        general_purpose::URL_SAFE_NO_PAD.encode(payload_json.as_bytes()),
        general_purpose::URL_SAFE_NO_PAD.encode(signature)
    );

    (collection_name, token)
}

fn collection_from_session(session_id: &str, config: &AppConfig) -> Result<String, ApiTokenError> {
    let mut parts = session_id.split('.');
    let payload_b64 = parts.next().ok_or(ApiTokenError::Invalid)?;
    let signature_b64 = parts.next().ok_or(ApiTokenError::Invalid)?;
    if parts.next().is_some() {
        return Err(ApiTokenError::Invalid);
    }

    let payload_bytes = general_purpose::URL_SAFE_NO_PAD
        .decode(payload_b64)
        .map_err(|_| ApiTokenError::Invalid)?;
    let provided_signature = general_purpose::URL_SAFE_NO_PAD
        .decode(signature_b64)
        .map_err(|_| ApiTokenError::Invalid)?;

    let mut mac =
        HmacSha256::new_from_slice(&config.session_secret).map_err(|_| ApiTokenError::Invalid)?;
    mac.update(&payload_bytes);
    mac.verify_slice(&provided_signature)
        .map_err(|_| ApiTokenError::Invalid)?;

    let payload: SessionPayload =
        serde_json::from_slice(&payload_bytes).map_err(|_| ApiTokenError::Invalid)?;
    if payload.exp < Utc::now().timestamp() {
        return Err(ApiTokenError::Expired);
    }

    Ok(payload.collection)
}
