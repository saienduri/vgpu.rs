use core::time::Duration;
use std::env;
use std::fs;

use api_types::PodInfoResponse;
use error_stack::{Report, ResultExt};
use reqwest::blocking::Client;

const SERVICE_ACCOUNT_TOKEN_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/token";
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Debug, Clone)]
pub struct PodConfig {
    pub gpu_uuids: Vec<String>,
    pub isolation: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Failed to read service account token")]
    TokenRead,
    #[error("Service account token is empty")]
    EmptyToken,
    #[error("HTTP request failed")]
    HttpRequest,
    #[error("Invalid response from hypervisor")]
    InvalidResponse,
    #[error("Missing required data in response")]
    MissingData,
    #[error("JSON parsing failed")]
    JsonParsing,
}

#[tracing::instrument(skip(hypervisor_ip, hypervisor_port), fields(url))]
pub fn get_worker_config(
    hypervisor_ip: impl AsRef<str>,
    hypervisor_port: impl AsRef<str>,
) -> Result<PodConfig, Report<ConfigError>> {
    request_pod_info(hypervisor_ip.as_ref(), hypervisor_port.as_ref())
}

#[tracing::instrument(level = "debug", fields(container_pid))]
fn request_pod_info(
    hypervisor_ip: &str,
    hypervisor_port: &str,
) -> Result<PodConfig, Report<ConfigError>> {
    let token = read_service_account_token()?;
    let container_pid = std::process::id();
    let container_name = env::var("CONTAINER_NAME").unwrap_or_default();
    let pod_name = env::var("POD_NAME").unwrap_or_default();
    let pod_namespace = env::var("POD_NAMESPACE").unwrap_or_default();

    tracing::Span::current().record("container_pid", container_pid);

    let client = build_http_client();
    let url = format!("http://{hypervisor_ip}:{hypervisor_port}/api/v1/pod");

    tracing::debug!(url = %url, pod_name = %pod_name, pod_namespace = %pod_namespace, "Requesting pod information");

    let request_start = std::time::Instant::now();

    let mut request = client
        .get(&url)
        .bearer_auth(&token)
        .query(&[("container_pid", container_pid.to_string())]);

    if !container_name.is_empty() {
        request = request.query(&[("container_name", &container_name)]);
    }
    if !pod_name.is_empty() {
        request = request.query(&[("pod_name", &pod_name)]);
    }
    if !pod_namespace.is_empty() {
        request = request.query(&[("pod_namespace", &pod_namespace)]);
    }

    let response = request
        .send()
        .change_context(ConfigError::HttpRequest)
        .attach_with(|| format!("Failed to send GET request to {url}"))?;

    if !response.status().is_success() {
        return Err(Report::new(ConfigError::HttpRequest).attach(format!(
            "HTTP request failed with status: {}",
            response.status()
        )));
    }

    let pod_response: PodInfoResponse = response
        .json()
        .change_context(ConfigError::JsonParsing)
        .attach("Failed to parse PodInfoResponse")?;

    if !pod_response.success {
        return Err(Report::new(ConfigError::InvalidResponse)
            .attach(format!("Hypervisor API error: {}", pod_response.message)));
    }

    let pod_info = pod_response
        .pod_info
        .ok_or_else(|| Report::new(ConfigError::MissingData).attach("No pod data in response"))?;

    let request_duration = request_start.elapsed();

    tracing::info!(
        duration_ms = request_duration.as_millis(),
        gpu_count = pod_info.gpu_uuids.len(),
        compute_shard = pod_info.compute_shard,
        "Successfully retrieved pod configuration"
    );

    Ok(PodConfig {
        gpu_uuids: pod_info.gpu_uuids,
        isolation: pod_info.isolation,
    })
}

#[tracing::instrument(level = "debug")]
fn read_service_account_token() -> Result<String, Report<ConfigError>> {
    let token = fs::read_to_string(SERVICE_ACCOUNT_TOKEN_PATH)
        .change_context(ConfigError::TokenRead)
        .attach_with(|| format!("Failed to read token from {SERVICE_ACCOUNT_TOKEN_PATH}"))?;

    let token = token.trim();

    if token.is_empty() {
        return Err(Report::new(ConfigError::EmptyToken).attach("Service account token is empty"));
    }

    tracing::debug!("Successfully read service account token");

    Ok(token.to_string())
}

struct TimeoutConfig {
    request: Duration,
    connect: Duration,
}

impl TimeoutConfig {
    fn from_env() -> Self {
        Self {
            request: Self::parse_env_timeout(
                "HTTP_REQUEST_TIMEOUT",
                DEFAULT_REQUEST_TIMEOUT,
                "request",
            ),
            connect: Self::parse_env_timeout(
                "HTTP_CONNECT_TIMEOUT",
                DEFAULT_CONNECT_TIMEOUT,
                "connect",
            ),
        }
    }

    fn parse_env_timeout(env_var: &str, default: Duration, timeout_type: &str) -> Duration {
        match env::var(env_var).ok().and_then(|s| parse_duration(&s).ok()) {
            Some(duration) => {
                tracing::debug!(
                    timeout_type = timeout_type,
                    timeout_seconds = duration.as_secs(),
                    "Using custom {} timeout from environment",
                    timeout_type
                );
                duration
            }
            None => {
                if env::var(env_var).is_ok() {
                    tracing::warn!(
                        env_var = env_var,
                        timeout_type = timeout_type,
                        default_timeout_seconds = default.as_secs(),
                        "Failed to parse {}, using default",
                        env_var
                    );
                }
                default
            }
        }
    }
}

fn build_http_client() -> Client {
    let timeouts = TimeoutConfig::from_env();

    tracing::debug!(
        request_timeout_seconds = timeouts.request.as_secs(),
        connect_timeout_seconds = timeouts.connect.as_secs(),
        "Building HTTP client with timeouts"
    );

    Client::builder()
        .timeout(timeouts.request)
        .connect_timeout(timeouts.connect)
        .build()
        .expect("should build HTTP client")
}

pub fn get_hypervisor_config() -> Option<(String, String)> {
    let hypervisor_ip = env::var("HYPERVISOR_IP").ok()?;
    let hypervisor_port = env::var("HYPERVISOR_PORT").ok()?;
    Some((hypervisor_ip, hypervisor_port))
}

pub fn parse_duration(duration_str: &str) -> Result<Duration, Report<ConfigError>> {
    let duration_str = duration_str.trim();

    if duration_str.is_empty() {
        return Err(Report::new(ConfigError::InvalidResponse).attach("Duration string is empty"));
    }

    humantime::parse_duration(duration_str)
        .change_context(ConfigError::InvalidResponse)
        .attach_with(|| format!("Failed to parse duration: {duration_str}"))
}
