import re
from typing import List, Tuple, Set

class Tokenizer:
    def __init__(self):
        self.word_pattern = re.compile(r'\b\w+\b', re.UNICODE)
    
    def tokenize(self, text: str) -> List[Tuple[str, int]]:
        """Токенизация текста с возвращением позиций (в нижнем регистре)"""
        tokens = []
        for match in self.word_pattern.finditer(text.lower()):
            tokens.append((match.group(), match.start()))
        return tokens
    
    def tokenize_field(self, field_name: str, text: str) -> List[Tuple[str, int]]:
        """Токенизация поля документа - теперь без добавления поля в терм"""
        return [(token, pos) for pos, (token, _) in enumerate(self.tokenize(text))]
    
    def extract_field_from_term(self, term: str) -> Tuple[str, str]:
        """Извлечение поля из термина запроса"""
        if ':' in term:
            field, term_text = term.split(':', 1)
            return field.strip(), term_text.strip().lower()
        return None, term.strip().lower()