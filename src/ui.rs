use crate::gpu::luminance;
use std::io;
use std::io::Error;
use termion::raw::IntoRawMode;
use tui::backend::TermionBackend;
use tui::layout::{Constraint, Direction, Layout};
use tui::widgets::{BarChart, Block, Borders, Widget};
use tui::Terminal;
use std::sync::mpsc;

pub fn tuihisto() -> Result<(), Error> {
    let stdout = io::stdout().into_raw_mode()?;
    let backend = TermionBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let size = terminal.size()?;
    terminal.draw(|mut f| {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .margin(3)
            .constraints([Constraint::Percentage(100)].as_ref())
            .split(size);
        BarChart::default()
            .block(Block::default().title("Histogram").borders(Borders::ALL))
            .data()
    */
    })?;
    Ok(())
}