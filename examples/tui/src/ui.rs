use crate::app::{App, RenderBlock, ScrollState};
use ai_core::Role;
use ratatui::{
    prelude::*,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

pub fn ui(f: &mut Frame, app: &mut App) {
    let size = f.area();

    let status_line = Line::from(vec![
        Span::styled("Status: ", Style::default().fg(Color::DarkGray)),
        Span::styled(app.scroll_status_text(), status_style_for_scroll(app)),
    ]);

    let input_text = if app.is_loading {
        "Loading...".to_string()
    } else if let Some(err) = &app.error {
        format!("Error: {}", err)
    } else if app.input.is_empty() {
        "Type a message and press Enter...".to_string()
    } else {
        app.input.clone()
    };

    let status_height =
        wrapped_row_count(std::slice::from_ref(&status_line), size.width).clamp(1, 2) as u16;
    let input_inner_width = size.width.saturating_sub(2);
    let input_content_height =
        wrapped_row_count(&[Line::from(input_text.as_str())], input_inner_width).clamp(1, 4) as u16;
    let input_height = input_content_height.saturating_add(2);

    // Split layout: messages (top) + status + input (bottom)
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(status_height),
            Constraint::Length(input_height),
        ])
        .split(size);

    // Messages area
    let messages_block = Block::default()
        .title(" Chat ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Gray))
        .title_style(
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )
        .border_type(ratatui::widgets::BorderType::Rounded);

    let mut lines: Vec<Line<'static>> = Vec::new();
    for message in &app.messages {
        let (label, label_style, body_style) = role_style(message.role);
        lines.push(Line::from(vec![Span::styled(label, label_style)]));

        let mut hid_thinking = false;
        for block in &message.blocks {
            match block {
                RenderBlock::Text(text) => {
                    push_markdown_lines(&mut lines, text, body_style, false);
                }
                RenderBlock::Thinking(text) => {
                    if app.show_thinking {
                        lines.push(Line::from(vec![Span::styled(
                            "  [thinking]",
                            Style::default()
                                .fg(Color::DarkGray)
                                .add_modifier(Modifier::ITALIC),
                        )]));
                        push_markdown_lines(
                            &mut lines,
                            text,
                            Style::default().fg(Color::Gray),
                            true,
                        );
                    } else {
                        hid_thinking = true;
                    }
                }
                RenderBlock::ToolUse { name, args } => {
                    lines.push(Line::from(vec![
                        Span::styled(
                            "  [tool call] ",
                            Style::default()
                                .fg(Color::LightCyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(name.clone(), Style::default().fg(Color::Cyan)),
                    ]));
                    push_markdown_lines(
                        &mut lines,
                        &format!("args: {}", args),
                        Style::default().fg(Color::Cyan),
                        true,
                    );
                }
                RenderBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    let tag = if *is_error {
                        "  [tool error] "
                    } else {
                        "  [tool result] "
                    };
                    let tag_style = if *is_error {
                        Style::default()
                            .fg(Color::LightRed)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default()
                            .fg(Color::LightGreen)
                            .add_modifier(Modifier::BOLD)
                    };
                    lines.push(Line::from(vec![
                        Span::styled(tag, tag_style),
                        Span::styled(tool_use_id.clone(), Style::default().fg(Color::Gray)),
                    ]));
                    push_markdown_lines(
                        &mut lines,
                        content,
                        if *is_error {
                            Style::default().fg(Color::Red)
                        } else {
                            Style::default().fg(Color::Green)
                        },
                        true,
                    );
                }
            }
        }

        if hid_thinking {
            lines.push(Line::from(vec![Span::styled(
                "  [thinking hidden: Ctrl+T to toggle]",
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            )]));
        }
        lines.push(Line::from(""));
    }

    if lines.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            "Welcome to ai-core TUI chat",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )]));
        lines.push(Line::from(vec![Span::styled(
            "Type in the Input box below and press Enter.",
            Style::default().fg(Color::Gray),
        )]));
        lines.push(Line::from(vec![Span::styled(
            "Ctrl+T toggles thinking, Ctrl+L clears history, Ctrl+C exits.",
            Style::default().fg(Color::Gray),
        )]));
    }

    // Account for wrapped text, not just logical lines, so live-follow stays pinned
    // to the real bottom of long assistant responses.
    let max_scroll = wrapped_line_count(&lines, chunks[0].width.saturating_sub(2))
        .saturating_sub(chunks[0].height.saturating_sub(2) as usize)
        .min(u16::MAX as usize) as u16;
    app.set_message_scroll_limit(max_scroll);

    let messages = Paragraph::new(lines)
        .block(messages_block)
        .wrap(Wrap { trim: false });

    let messages = messages.scroll((app.message_scroll_offset(), 0));

    f.render_widget(messages, chunks[0]);

    let status = Paragraph::new(status_line)
        .style(Style::default().fg(Color::Gray))
        .wrap(Wrap { trim: false });

    f.render_widget(status, chunks[1]);

    let input_block = Block::default()
        .title(" Input (Ctrl+C exit, Ctrl+L clear, Ctrl+T thinking) ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Gray))
        .title_style(
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )
        .border_type(ratatui::widgets::BorderType::Rounded);

    let input = Paragraph::new(input_text)
        .block(input_block)
        .wrap(Wrap { trim: false })
        .style(if app.is_loading {
            Style::default().fg(Color::Yellow)
        } else if app.error.is_some() {
            Style::default().fg(Color::LightRed)
        } else if app.input.is_empty() {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default().fg(Color::White)
        });

    f.render_widget(input, chunks[2]);

    // Show cursor in input area if not loading
    if !app.is_loading {
        let cursor = input_cursor_position(chunks[2], &app.input, input_content_height);
        f.set_cursor_position(cursor);
    }
}

fn status_style_for_scroll(app: &App) -> Style {
    match app.scroll_state {
        ScrollState::FollowLatest => Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD),
        ScrollState::Manual {
            pending_new_content: true,
            ..
        } => Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
        ScrollState::Manual {
            pending_new_content: false,
            ..
        } => Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    }
}

fn role_style(role: Role) -> (&'static str, Style, Style) {
    match role {
        Role::User => (
            "You",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
            Style::default().fg(Color::White),
        ),
        Role::Assistant => (
            "AI",
            Style::default()
                .fg(Color::LightBlue)
                .add_modifier(Modifier::BOLD),
            Style::default().fg(Color::White),
        ),
        Role::Tool => (
            "Tool",
            Style::default()
                .fg(Color::LightMagenta)
                .add_modifier(Modifier::BOLD),
            Style::default().fg(Color::Gray),
        ),
    }
}

fn push_markdown_lines(lines: &mut Vec<Line<'static>>, text: &str, base: Style, indented: bool) {
    let indent = if indented { "    " } else { "  " };
    let mut in_code_fence = false;

    for raw_line in text.lines() {
        let trimmed = raw_line.trim_start();

        if trimmed.starts_with("```") {
            in_code_fence = !in_code_fence;
            lines.push(Line::from(vec![
                Span::styled(indent, base),
                Span::styled(
                    "```",
                    Style::default()
                        .fg(Color::LightCyan)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
            continue;
        }

        if in_code_fence {
            lines.push(Line::from(vec![
                Span::styled(indent, base),
                Span::styled(raw_line.to_string(), Style::default().fg(Color::Cyan)),
            ]));
            continue;
        }

        if trimmed.is_empty() {
            lines.push(Line::from(""));
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("# ") {
            lines.push(Line::from(vec![
                Span::styled(indent, base),
                Span::styled(
                    rest.to_string(),
                    base.add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
                ),
            ]));
            continue;
        }

        if let Some(rest) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
        {
            let mut spans = vec![
                Span::styled(indent, base),
                Span::styled("• ", base.add_modifier(Modifier::BOLD)),
            ];
            spans.extend(inline_code_spans(rest, base));
            lines.push(Line::from(spans));
            continue;
        }

        if let Some((prefix, rest)) = ordered_prefix(trimmed) {
            let mut spans = vec![
                Span::styled(indent, base),
                Span::styled(prefix, base.add_modifier(Modifier::BOLD)),
            ];
            spans.extend(inline_code_spans(rest, base));
            lines.push(Line::from(spans));
            continue;
        }

        let mut spans = vec![Span::styled(indent, base)];
        spans.extend(inline_code_spans(raw_line, base));
        lines.push(Line::from(spans));
    }
}

fn ordered_prefix(line: &str) -> Option<(String, &str)> {
    let mut digits = String::new();
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.peek() {
        if c.is_ascii_digit() {
            digits.push(*c);
            chars.next();
        } else {
            break;
        }
    }

    if digits.is_empty() {
        return None;
    }

    if chars.next() != Some('.') || chars.next() != Some(' ') {
        return None;
    }

    let consumed = digits.len() + 2;
    Some((format!("{}. ", digits), &line[consumed..]))
}

fn inline_code_spans(text: &str, base: Style) -> Vec<Span<'static>> {
    let mut spans = Vec::new();

    for (index, part) in text.split('`').enumerate() {
        if part.is_empty() {
            continue;
        }

        if index % 2 == 1 {
            spans.push(Span::styled(
                part.to_string(),
                Style::default()
                    .fg(Color::LightCyan)
                    .add_modifier(Modifier::BOLD),
            ));
        } else {
            spans.push(Span::styled(part.to_string(), base));
        }
    }

    spans
}

fn wrapped_line_count<'a>(lines: &[Line<'a>], width: u16) -> usize {
    if width == 0 {
        return 0;
    }

    let width = width as usize;
    lines
        .iter()
        .map(|line| {
            let line_width = line.width();
            if line_width == 0 {
                1
            } else {
                line_width.div_ceil(width)
            }
        })
        .sum()
}

fn wrapped_row_count<'a>(lines: &[Line<'a>], width: u16) -> usize {
    wrapped_line_count(lines, width)
}

fn input_cursor_position(area: Rect, input: &str, content_height: u16) -> (u16, u16) {
    let inner_width = area.width.saturating_sub(2).max(1);
    let total_width = Line::from(input).width() as u16;
    let wrapped_rows = total_width / inner_width;
    let max_cursor_row = content_height.saturating_sub(1);
    let cursor_row = wrapped_rows.min(max_cursor_row);
    let cursor_col = if wrapped_rows > max_cursor_row {
        inner_width.saturating_sub(1)
    } else {
        total_width % inner_width
    };

    (
        area.x.saturating_add(1).saturating_add(cursor_col),
        area.y.saturating_add(1).saturating_add(cursor_row),
    )
}

#[cfg(test)]
mod tests {
    use super::{input_cursor_position, push_markdown_lines, wrapped_row_count};
    use ratatui::{prelude::Rect, style::Style, text::Line};

    #[test]
    fn wrapped_line_count_counts_visual_rows() {
        let lines = vec![Line::from("abcdefghij"), Line::from("klm")];

        assert_eq!(wrapped_row_count(&lines, 10), 2);
        assert_eq!(wrapped_row_count(&lines, 4), 4);
    }

    #[test]
    fn wrapped_row_count_handles_blank_lines_code_blocks_and_narrow_widths() {
        let mut lines = Vec::new();
        push_markdown_lines(
            &mut lines,
            "long assistant message\n\n```rust\nfn main() { println!(\"hi\"); }\n```\nfinal line",
            Style::default(),
            true,
        );

        assert_eq!(wrapped_row_count(&lines, 80), 6);
        assert!(wrapped_row_count(&lines, 8) > wrapped_row_count(&lines, 80));
    }

    #[test]
    fn input_cursor_position_tracks_wrapped_text() {
        let area = Rect::new(2, 3, 12, 4);

        assert_eq!(input_cursor_position(area, "abcd", 4), (7, 4));
        assert_eq!(input_cursor_position(area, "abcdefghijk", 4), (4, 5));
    }

    #[test]
    fn input_cursor_position_clamps_when_content_is_truncated() {
        let area = Rect::new(0, 0, 8, 3);

        assert_eq!(input_cursor_position(area, "abcdefghijklmnop", 1), (6, 1));
    }
}
