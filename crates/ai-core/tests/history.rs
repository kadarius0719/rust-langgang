//! Integration tests for conversation memory: `ChatHistory` + `ChatStore`.

use ai_core::{ChatHistory, ChatStore, InMemoryChatStore, Message, Role};

#[test]
fn history_builds_a_request_with_its_messages() {
    let mut history = ChatHistory::new();
    history
        .user("hello")
        .assistant("hi there")
        .user("how are you?");

    assert_eq!(history.len(), 3);

    let request = history
        .to_request("llama3.1")
        .system("be brief")
        .build()
        .unwrap();
    assert_eq!(request.messages.len(), 3);
    assert_eq!(request.messages[0].role, Role::User);
    assert_eq!(request.messages[0].text(), "hello");
    assert_eq!(request.system.as_deref(), Some("be brief"));
}

#[test]
fn record_response_appends_assistant_message() {
    use ai_core::ChatResponse;

    let mut history = ChatHistory::new();
    history.user("ping");
    let response = ChatResponse::from_message(Message::assistant("pong"));
    history.record_response(&response);

    assert_eq!(history.len(), 2);
    assert_eq!(history.last().unwrap().role, Role::Assistant);
    assert_eq!(history.last().unwrap().text(), "pong");
}

#[tokio::test]
async fn store_appends_loads_and_clears() {
    let store = InMemoryChatStore::new();

    store.append("s1", Message::user("hi")).await.unwrap();
    store
        .append_many(
            "s1",
            vec![Message::assistant("hello"), Message::user("bye")],
        )
        .await
        .unwrap();

    let loaded = store.load("s1").await.unwrap();
    assert_eq!(loaded.len(), 3);
    assert_eq!(loaded[0].text(), "hi");
    assert_eq!(loaded[2].text(), "bye");

    store.clear("s1").await.unwrap();
    assert!(store.load("s1").await.unwrap().is_empty());
}

#[tokio::test]
async fn store_save_replaces_and_isolates_sessions() {
    let store = InMemoryChatStore::new();

    store.append("a", Message::user("one")).await.unwrap();
    store.append("b", Message::user("two")).await.unwrap();

    // save() replaces a session wholesale.
    store.save("a", vec![Message::user("fresh")]).await.unwrap();

    assert_eq!(store.load("a").await.unwrap().len(), 1);
    assert_eq!(store.load("a").await.unwrap()[0].text(), "fresh");
    // Other session untouched.
    assert_eq!(store.load("b").await.unwrap()[0].text(), "two");
    assert_eq!(store.session_count(), 2);
}

#[tokio::test]
async fn history_round_trips_through_a_store() {
    let store = InMemoryChatStore::new();

    let mut history = ChatHistory::new();
    history.user("remember me").assistant("done");
    store
        .save("session-1", history.clone().into_messages())
        .await
        .unwrap();

    // Resume in a "new process".
    let resumed = ChatHistory::from_messages(store.load("session-1").await.unwrap());
    assert_eq!(resumed, history);
    assert_eq!(resumed.messages().len(), 2);
}
