use crate::graph::{JsonGraph, Link, Node};
use crate::local_embeddings::Embeddings;
use rocket::serde::{Deserialize, Serialize};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode};
use sqlx::{Connection, Row, SqliteConnection};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::path::{Path, PathBuf};

pub const EMBEDDINGS_SIZE: i32 = 1024; // multilingual-e5-large-instruct or qwen3
pub const EMBEDDINGS_DISTANCE: f32 = 0.6;
const GRAPH_SCORE_THRESHOLD: f32 = 0.65;

#[derive(Debug, Serialize, Deserialize)]
pub struct Document {
    pub id: i64,
    pub title: String,
    pub topic: Option<String>,
    pub text: String,
    pub timestamp: i64,
    pub score: i32,
}

#[derive(Debug, Clone)]
pub struct PendingTopic {
    pub group_id: u32,
    pub doc_titles: Vec<String>,
    pub doc_count: usize,
}

pub struct GraphComputation {
    pub graph: JsonGraph,
    pub pending_topics: Vec<PendingTopic>,
}

pub struct DataStoreClient {
    collection: String,
    collection_id: i64,
    embeddings_url: String,
    db_path: PathBuf,
}

fn extract_google_id(collection: &str) -> String {
    if collection.starts_with('G') && collection.ends_with('Z') && collection.len() > 2 {
        collection[1..collection.len() - 1].to_string()
    } else {
        collection.to_string()
    }
}

impl DataStoreClient {
    pub fn collection_id(&self) -> i64 {
        self.collection_id
    }

    pub async fn pending_topic_count(&self) -> Result<i64, Box<dyn Error>> {
        let mut conn = Self::connect_to_path(&self.db_path).await?;
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM topics WHERE collection_id = ? AND status = 'pending'",
        )
        .bind(self.collection_id)
        .fetch_one(&mut conn)
        .await
        .unwrap_or(0);
        Ok(count)
    }

    pub fn db_path(&self) -> PathBuf {
        self.db_path.clone()
    }

    pub async fn new(
        collection: &str,
        embeddings_url: &str,
        data_dir: &Path,
    ) -> Result<DataStoreClient, Box<dyn Error>> {
        let google_id = extract_google_id(collection);
        let suffix_len = google_id.len().min(3);
        let shard = &google_id[google_id.len() - suffix_len..];
        let shard_dir = data_dir.join(shard);
        std::fs::create_dir_all(&shard_dir)?;
        let db_path = shard_dir.join(format!("{}.db", google_id));

        let mut conn = Self::connect_to_path(&db_path).await?;
        Self::initialize_schema(&mut conn).await?;

        let collection_id: i64 =
            match sqlx::query_scalar("SELECT id FROM collections where name = ?")
                .bind(collection.to_string())
                .fetch_one(&mut conn)
                .await
            {
                Ok(id) => id,
                Err(_) => {
                    sqlx::query_scalar("INSERT INTO collections (name) values ( ? ) returning id")
                        .bind(collection.to_string())
                        .fetch_one(&mut conn)
                        .await?
                }
            };

        Ok(DataStoreClient {
            collection: collection.to_string(),
            collection_id,
            embeddings_url: embeddings_url.to_string(),
            db_path,
        })
    }

    pub async fn connect_to_path(path: &Path) -> Result<SqliteConnection, sqlx::Error> {
        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .foreign_keys(true);
        SqliteConnection::connect_with(&options).await
    }

    async fn initialize_schema(conn: &mut SqliteConnection) -> Result<(), sqlx::Error> {
        sqlx::query("PRAGMA foreign_keys = ON;")
            .execute(&mut *conn)
            .await?;
        sqlx::query("CREATE TABLE IF NOT EXISTS collections (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE);")
            .execute(&mut *conn)
            .await?;
        sqlx::query("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY AUTOINCREMENT, collection_id INTEGER,  title TEXT NOT NULL, topic TEXT, topic_group INTEGER, text TEXT, timestamp INTEGER, topic_dirty INTEGER NOT NULL DEFAULT 0, FOREIGN KEY (collection_id) REFERENCES collections (id));")
            .execute(&mut *conn)
            .await?;
        sqlx::query("CREATE TABLE IF NOT EXISTS topics (collection_id INTEGER NOT NULL, group_id INTEGER NOT NULL, label TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', doc_count INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (collection_id, group_id), FOREIGN KEY (collection_id) REFERENCES collections (id));")
            .execute(&mut *conn)
            .await?;
        let columns = sqlx::query("PRAGMA table_info(documents)")
            .fetch_all(&mut *conn)
            .await?;
        let has_topic = columns.iter().any(|row| {
            let name: String = row.get("name");
            name.eq_ignore_ascii_case("topic")
        });
        if !has_topic {
            sqlx::query("ALTER TABLE documents ADD COLUMN topic TEXT")
                .execute(&mut *conn)
                .await?;
        }
        let has_topic_group = columns.iter().any(|row| {
            let name: String = row.get("name");
            name.eq_ignore_ascii_case("topic_group")
        });
        if !has_topic_group {
            sqlx::query("ALTER TABLE documents ADD COLUMN topic_group INTEGER")
                .execute(&mut *conn)
                .await?;
        }
        let has_topic_dirty = columns.iter().any(|row| {
            let name: String = row.get("name");
            name.eq_ignore_ascii_case("topic_dirty")
        });
        if !has_topic_dirty {
            sqlx::query("ALTER TABLE documents ADD COLUMN topic_dirty INTEGER NOT NULL DEFAULT 0")
                .execute(&mut *conn)
                .await?;
        }
        let topic_columns = sqlx::query("PRAGMA table_info(topics)")
            .fetch_all(&mut *conn)
            .await?;
        let has_doc_count = topic_columns.iter().any(|row| {
            let name: String = row.get("name");
            name.eq_ignore_ascii_case("doc_count")
        });
        if !has_doc_count {
            sqlx::query("ALTER TABLE topics ADD COLUMN doc_count INTEGER NOT NULL DEFAULT 0")
                .execute(&mut *conn)
                .await?;
        }
        Ok(())
    }

    pub async fn check_embeddings(&self) -> Result<(), Box<dyn Error>> {
        let mut conn = Self::connect_to_path(&self.db_path).await?;
        let query = format!("CREATE VIRTUAL TABLE IF NOT EXISTS embeddings_{} USING vec0(id INTEGER PRIMARY KEY, embedding float[{:}])", &self.collection, EMBEDDINGS_SIZE);
        sqlx::query(&query).execute(&mut conn).await?;

        Ok(())
    }

    pub async fn list_recent(&self, length: u32) -> Result<Vec<Document>, Box<dyn Error>> {
        let mut conn = Self::connect_to_path(&self.db_path).await?;
        let mut docs = vec![];

        let rows = sqlx::query(
            "SELECT id, title, topic, text, timestamp FROM documents WHERE collection_id = ? ORDER BY timestamp DESC LIMIT ?",
        )
        .bind(self.collection_id)
        .bind(length as i64)
        .fetch_all(&mut conn)
        .await?;

        for r in rows {
            let doc_id: i64 = match r.get::<Option<i64>, _>("id") {
                Some(id) => id,
                None => continue,
            };
            docs.push(Document {
                id: doc_id,
                title: r.get::<String, _>("title"),
                topic: r.get::<Option<String>, _>("topic"),
                text: r.get::<Option<String>, _>("text").unwrap_or_default(),
                timestamp: r.get::<Option<i64>, _>("timestamp").unwrap_or_default(),
                score: 0,
            });
        }

        Ok(docs)
    }

    pub async fn list_related(
        &self,
        input: &str,
        length: u32,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let mut conn = Self::connect_to_path(&self.db_path).await?;
        let embeddings = Embeddings::new(&self.embeddings_url);
        let embedding_vecs = embeddings.fetch(input).await?;

        info!(
            "Embedding: [ {:2.4}; {:}",
            embedding_vecs[0],
            embedding_vecs.len()
        );

        let sub_query = format!("SELECT id, distance FROM embeddings_{} WHERE embedding MATCH ? AND distance > 0.0 AND distance < {:} ORDER BY distance LIMIT {:}", self.collection, EMBEDDINGS_DISTANCE, length);
        let rows = sqlx::query(&sub_query)
            .bind(format!("{:?}", embedding_vecs))
            .fetch_all(&mut conn)
            .await?;

        let mut docs = vec![];

        for r in rows {
            let id: i64 = r.get("id");
            let distance: f32 = r.get("distance");
            info!("id: {:}, distance: {:1.2}", id, distance);
            let score = ((1.0 - distance) * 100.0) as i32;

            let doc =
                sqlx::query("SELECT title, topic, text, timestamp FROM documents WHERE id = ?")
                    .bind(id)
                    .fetch_one(&mut conn)
                    .await?;

            docs.push(Document {
                id,
                title: doc.get("title"),
                topic: doc.get("topic"),
                text: doc.get::<Option<String>, _>("text").unwrap_or_default(),
                timestamp: doc.get::<Option<i64>, _>("timestamp").unwrap_or_default(),
                score,
            });
        }

        Ok(docs)
    }

    pub async fn retrieve_doc(&self, id: i64) -> Result<Document, Box<dyn Error>> {
        let mut conn = Self::connect_to_path(&self.db_path).await?;
        let row = sqlx::query(
            "SELECT title, topic, text, timestamp FROM documents WHERE id = ? AND collection_id = ?",
        )
        .bind(id)
        .bind(self.collection_id)
        .fetch_one(&mut conn)
        .await?;

        Ok(Document {
            id,
            title: row.get("title"),
            topic: row.get("topic"),
            text: match row.get("text") {
                Some(s) => s,
                None => "".to_string(),
            },
            timestamp: row.get("timestamp"),
            score: 0,
        })
    }

    pub async fn delete_doc(&self, id: i64) -> Result<(), Box<dyn Error>> {
        let mut conn = Self::connect_to_path(&self.db_path).await?;
        // Delete document
        sqlx::query("delete from documents where id = ? and collection_id = ?")
            .bind(id)
            .bind(self.collection_id)
            .execute(&mut conn)
            .await?;

        // delete embeddings
        let query = format!("delete from embeddings_{} where id = ?", &self.collection);
        sqlx::query(&query).bind(id).execute(&mut conn).await?;

        Ok(())
    }

    pub async fn save_document(
        &self,
        doc_id: Option<i64>,
        ciphertext: &str,
        plaintext: &str,
        title: &str,
    ) -> Result<Document, Box<dyn Error>> {
        let mut conn = Self::connect_to_path(&self.db_path).await?;
        let embeddings = Embeddings::new(&self.embeddings_url);
        let embedding_vecs = embeddings.fetch(plaintext).await?;

        let timestamp = chrono::offset::Utc::now().timestamp_millis() / 100;

        let id = if let Some(existing_id) = doc_id {
            sqlx::query("UPDATE documents SET title = ?, text = ?, timestamp = ?, topic = NULL, topic_group = NULL, topic_dirty = 1 WHERE id = ? AND collection_id = ?")
                .bind(title)
                .bind(ciphertext.to_string())
                .bind(timestamp)
                .bind(existing_id)
                .bind(self.collection_id)
                .execute(&mut conn)
                .await?;

            let delete_query = format!("delete from embeddings_{} where id = ?", &self.collection);
            sqlx::query(&delete_query)
                .bind(existing_id)
                .execute(&mut conn)
                .await?;
            existing_id
        } else {
            sqlx::query_scalar("insert into documents (collection_id, title, topic, topic_group, text, timestamp, topic_dirty) values (?, ?, NULL, NULL, ?, ?, 1) returning id")
                .bind(self.collection_id)
                .bind(title)
                .bind(ciphertext.to_string())
                .bind(timestamp)
                .fetch_one(&mut conn)
                .await?
        };

        // insert embeddings
        let query = format!(
            "insert into embeddings_{} (id, embedding) values (?, ?)",
            &self.collection
        );
        sqlx::query(&query)
            .bind(id)
            .bind(format!("{:?}", embedding_vecs))
            .execute(&mut conn)
            .await?;

        Ok(Document {
            id,
            title: title.to_string(),
            topic: None,
            timestamp,
            text: ciphertext.to_string(),
            score: 0,
        })
    }

    pub async fn list_graph(&self) -> Result<GraphComputation, Box<dyn Error>> {
        let mut conn = Self::connect_to_path(&self.db_path).await?;
        let limit: u64 = 4;
        let mut graph = JsonGraph {
            nodes: vec![],
            links: vec![],
            pending_topics: 0,
        };
        let mut doc_titles: HashMap<String, String> = HashMap::new();

        let docs = sqlx::query(
            "SELECT id, title, topic, topic_dirty FROM documents WHERE collection_id = ?",
        )
        .bind(self.collection_id)
        .fetch_all(&mut conn)
        .await?;

        let mut dirty_docs: HashMap<String, bool> = HashMap::new();
        for d in &docs {
            let doc_id = match d.get::<Option<i64>, _>("id") {
                Some(id) => id,
                None => continue,
            };
            let title: String = d.get::<String, _>("title");
            let dirty_flag = d
                .get::<Option<i64>, _>("topic_dirty")
                .map(|v| v != 0)
                .unwrap_or(false);
            dirty_docs.insert(doc_id.to_string(), dirty_flag);
            doc_titles.insert(doc_id.to_string(), title.clone());
            let node = Node {
                id: doc_id.to_string(),
                name: title,
                group: 0,
                topic: d.get::<Option<String>, _>("topic"),
            };

            let sub_query = format!("SELECT id, distance FROM embeddings_{} WHERE embedding MATCH (SELECT embedding FROM embeddings_{} WHERE id = ?) AND distance < {:} ORDER BY distance LIMIT {:}", self.collection, self.collection, GRAPH_SCORE_THRESHOLD, limit);
            let rows = sqlx::query(&sub_query)
                .bind(doc_id)
                .fetch_all(&mut conn)
                .await?;

            for i in rows {
                let score: f32 = i.get("distance");
                if score < GRAPH_SCORE_THRESHOLD {
                    let id: i64 = i.get("id");
                    graph.links.push(Link {
                        source: node.id.to_string(),
                        target: id.to_string(),
                    });
                }
            }

            graph.nodes.push(node);
        }

        let group_mapping = graph.assign_groups();
        let mut group_docs: HashMap<u32, Vec<String>> = HashMap::new();
        let mut group_counts: HashMap<u32, usize> = HashMap::new();
        let mut dirty_groups: HashSet<u32> = HashSet::new();
        for (doc_id, group_id) in &group_mapping {
            *group_counts.entry(*group_id).or_insert(0) += 1;
            if let Some(title) = doc_titles.get(doc_id) {
                group_docs.entry(*group_id).or_default().push(title.clone());
            }
            if dirty_docs.get(doc_id).copied().unwrap_or(false) {
                dirty_groups.insert(*group_id);
            }
        }

        let (label_map, pending_groups) = self
            .ensure_topic_records(&mut conn, &group_counts, &dirty_groups)
            .await?;
        self.persist_group_topics(&mut conn, &group_mapping, &label_map)
            .await?;

        for node in graph.nodes.iter_mut() {
            if let Some(group_id) = group_mapping.get(&node.id) {
                if let Some(label) = label_map.get(group_id) {
                    node.topic = Some(label.clone());
                }
            }
        }

        let pending_topics: Vec<PendingTopic> = pending_groups
            .into_iter()
            .map(|gid| PendingTopic {
                group_id: gid,
                doc_titles: group_docs.get(&gid).cloned().unwrap_or_default(),
                doc_count: *group_counts.get(&gid).unwrap_or(&0),
            })
            .collect();
        graph.pending_topics = pending_topics.len();

        Ok(GraphComputation {
            graph,
            pending_topics,
        })
    }

    async fn ensure_topic_records(
        &self,
        conn: &mut SqliteConnection,
        group_counts: &HashMap<u32, usize>,
        dirty_groups: &HashSet<u32>,
    ) -> Result<(HashMap<u32, String>, Vec<u32>), Box<dyn Error>> {
        let rows = sqlx::query(
            "SELECT group_id, label, status, doc_count FROM topics WHERE collection_id = ?",
        )
        .bind(self.collection_id)
        .fetch_all(&mut *conn)
        .await?;

        let mut label_map: HashMap<u32, String> = HashMap::new();
        let mut pending = HashSet::new();

        for row in rows {
            let gid: i64 = row.get("group_id");
            let label: String = row.get("label");
            let status: String = row.get("status");
            let existing_count: i64 = row.try_get("doc_count").unwrap_or(0);
            label_map.insert(gid as u32, label);

            if let Some(current_count) = group_counts.get(&(gid as u32)) {
                if existing_count as usize != *current_count {
                    sqlx::query(
                        "UPDATE topics SET status = 'pending', doc_count = ? WHERE collection_id = ? AND group_id = ?",
                    )
                    .bind(*current_count as i64)
                    .bind(self.collection_id)
                    .bind(gid)
                    .execute(&mut *conn)
                    .await?;
                    pending.insert(gid as u32);
                }
            }

            if dirty_groups.contains(&(gid as u32)) && status != "pending" {
                sqlx::query(
                    "UPDATE topics SET status = 'pending' WHERE collection_id = ? AND group_id = ?",
                )
                .bind(self.collection_id)
                .bind(gid)
                .execute(&mut *conn)
                .await?;
                pending.insert(gid as u32);
            } else if status == "pending" {
                pending.insert(gid as u32);
            }
        }

        for (gid, count) in group_counts {
            if !label_map.contains_key(gid) {
                let placeholder = format!("Topic {}", gid + 1);
                sqlx::query("INSERT INTO topics (collection_id, group_id, label, status, doc_count) VALUES (?, ?, ?, 'pending', ?)")
                    .bind(self.collection_id)
                    .bind(*gid as i64)
                    .bind(&placeholder)
                    .bind(*count as i64)
                    .execute(&mut *conn)
                    .await?;
                label_map.insert(*gid, placeholder);
                pending.insert(*gid);
            }
        }

        Ok((label_map, pending.into_iter().collect()))
    }

    async fn persist_group_topics(
        &self,
        conn: &mut SqliteConnection,
        group_mapping: &HashMap<String, u32>,
        label_map: &HashMap<u32, String>,
    ) -> Result<(), Box<dyn Error>> {
        for (doc_id_str, group_id) in group_mapping {
            if let Ok(doc_id) = doc_id_str.parse::<i64>() {
                if let Some(label) = label_map.get(group_id) {
                    sqlx::query("UPDATE documents SET topic = ?, topic_group = ?, topic_dirty = 0 WHERE id = ? AND collection_id = ?")
                        .bind(label)
                        .bind(*group_id as i64)
                        .bind(doc_id)
                        .bind(self.collection_id)
                        .execute(&mut *conn)
                        .await?;
                }
            }
        }
        Ok(())
    }
}
