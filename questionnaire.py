"""
Pre-test questionnaire UI for AX-CPT task
Simple Pygame-based form for collecting session metadata
"""
import pygame
from typing import Dict, Any, Optional


class PreTestQuestionnaire:
    """Simple questionnaire dialog for collecting session metadata."""
    
    def __init__(self, screen, bg_color=(0, 0, 0), text_color=(255, 255, 255)):
        """
        Initialize questionnaire.
        
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
        self.title_font = pygame.font.Font(None, 48)
        self.header_font = pygame.font.Font(None, 32)
        self.text_font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 24)
        
        # Questions structure
        self.questions = [
            {
                'section': 'ADHD Medication',
                'fields': [
                    {'key': 'adhd_med_taken', 'prompt': 'Taken today? (y/n)', 'type': 'yesno'},
                    {'key': 'hours_since_med', 'prompt': 'Hours since dose', 'type': 'number', 'depends_on': 'adhd_med_taken'}
                ]
            },
            {
                'section': 'Sleep & Energy',
                'fields': [
                    {'key': 'sleep_hours', 'prompt': 'Hours of sleep', 'type': 'number'},
                    {'key': 'mental_fatigue', 'prompt': 'Mental fatigue (1-10)', 'type': 'number'}
                ]
            },
            {
                'section': 'Substances',
                'fields': [
                    {'key': 'caffeine_hours_ago', 'prompt': 'Hours since caffeine (blank if none)', 'type': 'number'}
                ]
            },
            {
                'section': 'Physical Activity',
                'fields': [
                    {'key': 'exercise_hours_ago', 'prompt': 'Hours since exercise (blank if none)', 'type': 'number'}
                ]
            },
            {
                'section': 'Mood',
                'fields': [
                    {'key': 'stress_level', 'prompt': 'Stress level (1-10)', 'type': 'number'}
                ]
            },
            {
                'section': 'Notes',
                'fields': [
                    {'key': 'notes', 'prompt': 'Free text notes', 'type': 'text'}
                ]
            }
        ]
    
    def show(self, timestamp: str) -> Optional[Dict[str, Any]]:
        """
        Display questionnaire and collect responses.
        
        Args:
            timestamp: Session timestamp
            
        Returns:
            Dictionary of responses, or None if skipped
        """
        responses = {'timestamp': timestamp}
        
        # Show intro screen
        if not self._show_intro():
            return None  # User pressed ESC to skip
        
        # Collect responses for each question
        for section_data in self.questions:
            section = section_data['section']
            
            for field in section_data['fields']:
                # Check if this field depends on a previous answer
                if 'depends_on' in field:
                    dep_key = field['depends_on']
                    if responses.get(dep_key, '').lower() not in ['y', 'yes']:
                        responses[field['key']] = ''
                        continue
                
                # Ask the question
                response = self._ask_question(section, field)
                
                if response is False:  # ESC pressed
                    return None
                
                responses[field['key']] = response if response else ''

        return responses

    def _show_intro(self) -> bool:
        """
        Show intro screen explaining the questionnaire.

        Returns:
            True if user wants to continue, False if ESC pressed
        """
        self.screen.fill(self.bg_color)

        # Title
        title = self.title_font.render("Pre-Test Questionnaire", True, self.text_color)
        title_rect = title.get_rect(center=(self.center[0], 100))
        self.screen.blit(title, title_rect)

        # Instructions
        instructions = [
            "This quick questionnaire helps track factors that may affect performance.",
            "",
            "All fields are optional - press ENTER to skip any question.",
            "",
            "Press ESC now to skip the entire questionnaire,",
            "or press SPACE to continue..."
        ]

        y = 200
        for line in instructions:
            text = self.text_font.render(line, True, self.text_color)
            text_rect = text.get_rect(center=(self.center[0], y))
            self.screen.blit(text, text_rect)
            y += 40

        pygame.display.flip()

        # Wait for response
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    elif event.key == pygame.K_SPACE:
                        return True

        return True

    def _ask_question(self, section: str, field: Dict[str, Any]) -> Any:
        """
        Ask a single question and get response.

        Args:
            section: Section name
            field: Field dictionary with key, prompt, type

        Returns:
            User's response, empty string if skipped, False if ESC pressed
        """
        input_text = ""
        active = True

        while active:
            self.screen.fill(self.bg_color)

            # Section header
            y = 80
            section_text = self.header_font.render(section, True, (100, 200, 255))
            section_rect = section_text.get_rect(center=(self.center[0], y))
            self.screen.blit(section_text, section_rect)

            # Question prompt
            y = 160
            prompt_text = self.text_font.render(field['prompt'], True, self.text_color)
            prompt_rect = prompt_text.get_rect(center=(self.center[0], y))
            self.screen.blit(prompt_text, prompt_rect)

            # Input box
            y = 240
            input_display = self.text_font.render(f"> {input_text}_", True, (255, 255, 100))
            input_rect = input_display.get_rect(center=(self.center[0], y))
            self.screen.blit(input_display, input_rect)

            # Help text
            y = 320
            help_text = self.small_font.render("ENTER to submit | ESC to skip questionnaire", True, (150, 150, 150))
            help_rect = help_text.get_rect(center=(self.center[0], y))
            self.screen.blit(help_text, help_rect)

            pygame.display.flip()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    elif event.key == pygame.K_RETURN:
                        # Validate and return
                        return self._validate_input(input_text, field['type'])
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        # Add character if valid
                        char = event.unicode
                        if char and self._is_valid_char(char, field['type']):
                            input_text += char

        return ""

    def _is_valid_char(self, char: str, field_type: str) -> bool:
        """Check if character is valid for field type."""
        if field_type == 'number':
            return char.isdigit() or char == '.'
        elif field_type == 'yesno':
            return char.lower() in 'yn'
        else:  # text
            return len(char) == 1 and char.isprintable()

    def _validate_input(self, text: str, field_type: str) -> Any:
        """Validate and convert input based on field type."""
        text = text.strip()

        if not text:
            return ''

        if field_type == 'number':
            try:
                # Try to convert to float
                return float(text)
            except ValueError:
                return ''
        elif field_type == 'yesno':
            return text.lower()
        else:  # text
            return text


