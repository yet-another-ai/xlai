use crate::request::GeminiChatRequest;
use xlai_core::{ChatContent, ChatRequest, MessageRole};

#[test]
fn serializes_basic_request() {
    let request = ChatRequest {
        model: None,
        system_prompt: None,
        messages: vec![
            xlai_core::ChatMessage {
                role: MessageRole::System,
                content: ChatContent::text("You are a helpful assistant."),
                tool_name: None,
                tool_call_id: None,
                metadata: Default::default(),
            },
            xlai_core::ChatMessage {
                role: MessageRole::User,
                content: ChatContent::text("Hello!"),
                tool_name: None,
                tool_call_id: None,
                metadata: Default::default(),
            },
        ],
        available_tools: Vec::new(),
        structured_output: None,
        metadata: Default::default(),
        temperature: None,
        max_output_tokens: None,
        reasoning_effort: None,
        retry_policy: None,
    };

    let gemini_req = GeminiChatRequest::from_core_request(request).unwrap();

    assert!(gemini_req.system_instruction.is_some());
    assert_eq!(gemini_req.contents.len(), 1);
    assert_eq!(gemini_req.contents[0].role, "user");
}
