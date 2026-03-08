"""
Progress bar UI component for displaying processing status
"""
import pygame
from typing import Optional


class ProgressBar:
    """Display a progress bar with percentage and status text."""
    
    def __init__(self, screen, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
        """
        Initialize progress bar.
        
        Args:
            screen: Pygame screen surface
            bg_color: Background color
            text_color: Text color
        """
        self.screen = screen
        self.bg_color = bg_color
        self.text_color = text_color
        self.width = screen.get_width()
        self.height = screen.get_height()
        self.center = (self.width // 2, self.height // 2)
        
        # Font sizes
        self.title_font = pygame.font.Font(None, 56)
        self.text_font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        
        # Progress bar dimensions
        self.bar_width = 600
        self.bar_height = 40
        self.bar_x = (self.width - self.bar_width) // 2
        self.bar_y = self.center[1] + 20
        
        # Colors
        self.bar_bg_color = (50, 50, 50)
        self.bar_fill_color = (0, 200, 100)
        self.bar_border_color = (100, 100, 100)
        
    def show(self, progress: float, status_text: str = "Processing...", 
             detail_text: Optional[str] = None):
        """
        Display progress bar.
        
        Args:
            progress: Progress value from 0.0 to 1.0
            status_text: Main status message
            detail_text: Optional detailed status (e.g., "Frame 100/500")
        """
        # Clamp progress to valid range
        progress = max(0.0, min(1.0, progress))
        
        # Clear screen
        self.screen.fill(self.bg_color)
        
        # Draw title
        title_surface = self.title_font.render(status_text, True, self.text_color)
        title_rect = title_surface.get_rect(center=(self.center[0], self.center[1] - 80))
        self.screen.blit(title_surface, title_rect)
        
        # Draw detail text if provided
        if detail_text:
            detail_surface = self.text_font.render(detail_text, True, self.text_color)
            detail_rect = detail_surface.get_rect(center=(self.center[0], self.center[1] - 30))
            self.screen.blit(detail_surface, detail_rect)
        
        # Draw progress bar background
        bar_bg_rect = pygame.Rect(self.bar_x, self.bar_y, self.bar_width, self.bar_height)
        pygame.draw.rect(self.screen, self.bar_bg_color, bar_bg_rect)
        
        # Draw progress bar fill
        fill_width = int(self.bar_width * progress)
        if fill_width > 0:
            bar_fill_rect = pygame.Rect(self.bar_x, self.bar_y, fill_width, self.bar_height)
            pygame.draw.rect(self.screen, self.bar_fill_color, bar_fill_rect)
        
        # Draw progress bar border
        pygame.draw.rect(self.screen, self.bar_border_color, bar_bg_rect, 2)
        
        # Draw percentage text
        percentage = int(progress * 100)
        percent_text = f"{percentage}%"
        percent_surface = self.text_font.render(percent_text, True, self.text_color)
        percent_rect = percent_surface.get_rect(center=(self.center[0], self.bar_y + self.bar_height + 40))
        self.screen.blit(percent_surface, percent_rect)
        
        # Draw instruction text
        instruction_text = "Please wait..."
        instruction_surface = self.small_font.render(instruction_text, True, self.text_color)
        instruction_rect = instruction_surface.get_rect(center=(self.center[0], self.bar_y + self.bar_height + 80))
        self.screen.blit(instruction_surface, instruction_rect)
        
        # Update display
        pygame.display.flip()
        
    def update(self, progress: float, current: int, total: int, 
               status_text: str = "Processing Frames"):
        """
        Update progress bar with current/total counts.
        
        Args:
            progress: Progress value from 0.0 to 1.0
            current: Current item number
            total: Total number of items
            status_text: Main status message
        """
        detail_text = f"{current}/{total} frames"
        self.show(progress, status_text, detail_text)
        
        # Process events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass  # Ignore quit during processing

