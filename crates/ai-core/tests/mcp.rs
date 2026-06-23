use ai_core::{load_mcp_tools, Error, McpClient, ToolBox};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn mcp_client_initializes_lists_and_calls_tools() {
    let server = MockServer::start().await;
    mount_initialize(&server).await;
    mount_list_tools(
        &server,
        serde_json::json!([
            {
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "inputSchema": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }
            }
        ]),
    )
    .await;
    mount_call_tool(
        &server,
        "get_weather",
        serde_json::json!({"city": "Paris"}),
        serde_json::json!({
            "content": [{ "type": "text", "text": "{\"city\":\"Paris\",\"temp_f\":72}" }]
        }),
    )
    .await;

    let client = McpClient::connect(server.uri()).await.unwrap();
    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "get_weather");
    assert_eq!(tools[0].description, "Get the weather for a city.");

    let result = client
        .call_tool("get_weather", serde_json::json!({"city": "Paris"}))
        .await
        .unwrap();
    assert_eq!(result["city"], "Paris");
    assert_eq!(result["temp_f"], 72);
}

#[tokio::test]
async fn mcp_jsonrpc_error_maps_to_tool_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/"))
        .and(body_partial_json(
            serde_json::json!({ "method": "initialize" }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": { "code": -32000, "message": "boom" }
        })))
        .mount(&server)
        .await;

    let err = McpClient::connect(server.uri()).await.err().unwrap();
    assert!(matches!(err, Error::Tool(_)));
    assert!(err.to_string().contains("boom"));
}

#[tokio::test]
async fn load_mcp_tools_namespaces_multiple_servers_and_preserves_descriptions() {
    let server_a = MockServer::start().await;
    let server_b = MockServer::start().await;
    mount_initialize(&server_a).await;
    mount_initialize(&server_b).await;
    mount_list_tools(
        &server_a,
        serde_json::json!([
            {
                "name": "get_weather",
                "description": "Weather from server A.",
                "inputSchema": { "type": "object" }
            }
        ]),
    )
    .await;
    mount_list_tools(
        &server_b,
        serde_json::json!([
            {
                "name": "get_weather",
                "description": "Weather from server B.",
                "inputSchema": { "type": "object" }
            }
        ]),
    )
    .await;
    mount_call_tool(
        &server_a,
        "get_weather",
        serde_json::json!({}),
        serde_json::json!({ "content": [{ "type": "text", "text": "{\"source\":\"a\"}" }] }),
    )
    .await;
    mount_call_tool(
        &server_b,
        "get_weather",
        serde_json::json!({}),
        serde_json::json!({ "content": [{ "type": "text", "text": "{\"source\":\"b\"}" }] }),
    )
    .await;

    let mut tools = ToolBox::new();
    let reports = load_mcp_tools(
        &mut tools,
        [("server-a", server_a.uri()), ("server b", server_b.uri())],
    )
    .await;

    assert_eq!(reports.len(), 2);
    assert!(reports.iter().all(|report| report.error.is_none()));
    assert!(tools.get("server_a_get_weather").is_some());
    assert!(tools.get("server_b_get_weather").is_some());

    let def_a = tools.get("server_a_get_weather").unwrap().def();
    assert!(def_a.description.contains("server_a"));
    assert!(def_a.description.contains("get_weather"));
    assert!(def_a.description.contains("Weather from server A."));

    let result_a = tools
        .invoke("server_a_get_weather", serde_json::json!({}))
        .await
        .unwrap();
    let result_b = tools
        .invoke("server_b_get_weather", serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(result_a["source"], "a");
    assert_eq!(result_b["source"], "b");
}

#[tokio::test]
async fn load_mcp_tools_keeps_successful_servers_when_one_fails() {
    let good = MockServer::start().await;
    mount_initialize(&good).await;
    mount_list_tools(
        &good,
        serde_json::json!([
            {
                "name": "ping",
                "description": "",
                "inputSchema": { "type": "object" }
            }
        ]),
    )
    .await;

    let mut tools = ToolBox::new();
    let reports = load_mcp_tools(
        &mut tools,
        vec![
            ("good", good.uri()),
            ("bad", "http://127.0.0.1:9".to_string()),
        ],
    )
    .await;

    assert_eq!(reports.len(), 2);
    assert!(reports
        .iter()
        .any(|report| report.label == "good" && report.error.is_none()));
    assert!(reports
        .iter()
        .any(|report| report.label == "bad" && report.error.is_some()));
    assert!(tools.get("good_ping").is_some());
}

async fn mount_initialize(server: &MockServer) {
    Mock::given(method("POST"))
        .and(path("/"))
        .and(body_partial_json(
            serde_json::json!({ "method": "initialize" }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "serverInfo": { "name": "mock-mcp", "version": "0.1.0" }
            }
        })))
        .mount(server)
        .await;

    Mock::given(method("POST"))
        .and(path("/"))
        .and(body_partial_json(
            serde_json::json!({ "method": "notifications/initialized" }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0",
            "result": {}
        })))
        .mount(server)
        .await;
}

async fn mount_list_tools(server: &MockServer, tools: serde_json::Value) {
    Mock::given(method("POST"))
        .and(path("/"))
        .and(body_partial_json(
            serde_json::json!({ "method": "tools/list" }),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "result": { "tools": tools }
        })))
        .mount(server)
        .await;
}

async fn mount_call_tool(
    server: &MockServer,
    name: &str,
    arguments: serde_json::Value,
    result: serde_json::Value,
) {
    Mock::given(method("POST"))
        .and(path("/"))
        .and(body_partial_json(serde_json::json!({
            "method": "tools/call",
            "params": { "name": name, "arguments": arguments }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "result": result
        })))
        .mount(server)
        .await;
}
