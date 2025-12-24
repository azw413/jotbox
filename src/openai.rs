use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug)]
pub struct ChatClient {
    client: HttpClient,
    api_base: String,
    api_key: Option<String>,
    model: String,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    enable_thinking: bool,
}

#[derive(Debug, Deserialize)]
#[serde(crate = "rocket::serde")]
pub struct Interaction {
    pub(crate) role: String,
    pub(crate) message: String,
}

#[derive(Debug, Deserialize)]
#[serde(crate = "rocket::serde")]
pub struct ChatInteraction {
    pub(crate) history: Vec<Interaction>,
    pub(crate) query: String,
    pub(crate) context: String,
    pub(crate) sources: Vec<SourceSummary>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(crate = "rocket::serde")]
pub struct SourceSummary {
    pub(crate) id: i64,
    pub(crate) title: String,
    pub(crate) score: i32,
}

#[derive(Debug, Error)]
#[error(transparent)]
pub enum OpenAIEmbeddingsError {
    #[error("Request failed: {0}")]
    Request(String),
}

impl ChatClient {
    pub fn new(
        api_base: &str,
        api_key: Option<String>,
        model: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        enable_thinking: bool,
    ) -> Self {
        Self {
            client: HttpClient::new(),
            api_base: api_base.trim_end_matches('/').to_string(),
            api_key,
            model: model.to_string(),
            max_tokens,
            temperature,
            top_p,
            top_k,
            enable_thinking,
        }
    }

    pub(crate) async fn chat(
        &self,
        interaction: &ChatInteraction,
        context: &str,
    ) -> Result<String, OpenAIEmbeddingsError> {
        let today = chrono::offset::Local::now()
            .format("%a %b %e %Y, %T")
            .to_string();

        let mut messages = vec![];
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: format!("You are a helpful and informal assistant that outputs in markdown. The date and time are {}. The input format is relevant documents followed by chat. Don't mention the documents specifically. Do not think and respond with confidence.", today),
        });
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: context.to_string(),
        });

        for i in &interaction.history {
            messages.push(ChatMessage {
                role: i.role.clone(),
                content: i.message.clone(),
            });
        }

        messages.push(ChatMessage {
            role: "user".to_string(),
            content: interaction.query.clone(),
        });

        let payload = ChatCompletionRequest {
            model: self.model.clone(),
            max_tokens: Some(self.max_tokens),
            temperature: Some(self.temperature),
            top_p: Some(self.top_p),
            top_k: Some(self.top_k),
            enable_thinking: Some(self.enable_thinking),
            messages,
        };

        let url = format!("{}/chat/completions", self.api_base);
        let mut request = self.client.post(url).json(&payload);
        if let Some(key) = &self.api_key {
            if !key.is_empty() {
                request = request.bearer_auth(key);
            }
        }

        let response = request
            .send()
            .await
            .map_err(|e| OpenAIEmbeddingsError::Request(e.to_string()))?;

        if !response.status().is_success() {
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(OpenAIEmbeddingsError::Request(text));
        }

        let raw_body = response
            .text()
            .await
            .map_err(|e| OpenAIEmbeddingsError::Request(e.to_string()))?;
        info!("LLM response: {}", raw_body);

        let completion: ChatCompletionResponse = serde_json::from_str(&raw_body).map_err(|e| {
            OpenAIEmbeddingsError::Request(format!(
                "Failed to parse LLM response: {e}; body: {raw_body}"
            ))
        })?;

        completion
            .choices
            .get(0)
            .map(|choice| {
                let content = choice.message.content.trim();
                if !content.is_empty() {
                    content.to_string()
                } else if let Some(reasoning) = choice
                    .message
                    .reasoning
                    .as_ref()
                    .filter(|r| !r.trim().is_empty())
                {
                    reasoning.trim().to_string()
                } else {
                    String::from("(LLM returned an empty response)")
                }
            })
            .ok_or_else(|| OpenAIEmbeddingsError::Request("Empty response from LLM".to_string()))
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_thinking: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChatChoiceMessage {
    role: String,
    content: String,
    #[serde(default)]
    reasoning: Option<String>,
}
