use anyhow::{anyhow, Context};
use std::collections::HashMap;

/// Simple client for HuggingFace's text-embedding-inference tool

pub struct Embeddings {
    url: String,
}

impl Embeddings {
    pub fn new(url: &str) -> Embeddings {
        Embeddings {
            url: url.to_string(),
        }
    }

    pub async fn fetch(&self, input: &str) -> Result<Vec<f32>, anyhow::Error> {
        let mut map = HashMap::new();
        map.insert("inputs", input.to_string());

        let client = reqwest::Client::new();
        let response = client
            .post(&self.url)
            .json(&map)
            .send()
            .await
            .context("Failed to call embeddings service")?;

        let status = response.status();
        let body = response
            .text()
            .await
            .context("Failed to read embeddings response body")?;

        if !status.is_success() {
            return Err(anyhow!(
                "Embeddings service returned {}: {}",
                status,
                body.trim()
            ));
        }

        let mut data: Vec<Vec<f32>> = serde_json::from_str(&body)
            .with_context(|| format!("Failed to parse embeddings response: {}", body))?;
        data.pop()
            .ok_or_else(|| anyhow!("Embeddings response missing vector"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datastore::EMBEDDINGS_SIZE;
    use std::env;
    use tokio::runtime::Runtime;

    #[test]
    fn test_embedding() {
        let url = match env::var("JOTBOX_EMBEDDINGS_URL") {
            Ok(u) => u,
            Err(_) => {
                eprintln!("Skipping embedding test because JOTBOX_EMBEDDINGS_URL is not set");
                return;
            }
        };
        let embedding = Embeddings::new(&url);
        let result = Runtime::new()
            .unwrap()
            .block_on(embedding.fetch("This is a test phrase for embeddings retrieval."));
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), EMBEDDINGS_SIZE as usize);
    }
}
