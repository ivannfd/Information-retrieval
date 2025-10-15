from typing import Dict, List, Set, Tuple, Any, DefaultDict
from collections import defaultdict
from .document import Document
from .tokenizer import Tokenizer

class InvertedIndex:
    def __init__(self):
        # Структура индекса: term -> field -> doc_id -> positions
        self.index: DefaultDict[str, DefaultDict[str, DefaultDict[str, List[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self.documents: Dict[str, Document] = {}
        self.tokenizer = Tokenizer()
        self.fields: Set[str] = set()
    
    def add_document(self, doc: Document):
        """Добавление документа в индекс с правильной нумерацией позиций"""
        self.documents[doc.id] = doc

        all_tokens = []
        current_position = 0
        
        for field_name, field_text in doc.fields.items():
            self.fields.add(field_name)
            tokens_with_positions = self.tokenizer.tokenize_field(field_name, field_text)

            for token, position in tokens_with_positions:
                all_tokens.append((token, field_name, current_position))
                current_position += 1

        for token, field_name, position in all_tokens:
            self.index[token][field_name][doc.id].append(position)
    
    def get_postings(self, term: str, field: str = None) -> Dict[str, List[int]]:
        """Получение постингов для термина с учетом поля"""
        term = term.lower()
        if field:
            return dict(self.index.get(term, {}).get(field, {}))
        else:
            result = {}
            for field_name in self.index.get(term, {}):
                for doc_id, positions in self.index[term][field_name].items():
                    if doc_id in result:
                        result[doc_id].extend(positions)
                    else:
                        result[doc_id] = positions.copy()
            return result
    
    def get_document_positions(self, doc_id: str, field: str, term: str) -> List[int]:
        """Получение позиций термина в конкретном поле документа"""
        term = term.lower()
        return self.index.get(term, {}).get(field, {}).get(doc_id, [])
    
    def search_term(self, term: str, field: str = None) -> Set[str]:
        """Поиск документов по термину с учетом поля"""
        term = term.lower()
        postings = self.get_postings(term, field)
        return set(postings.keys())
    
    def get_term_fields(self, term: str) -> Set[str]:
        """Получение полей, в которых встречается термин"""
        term = term.lower()
        return set(self.index.get(term, {}).keys())

class SearchEngine:
    def __init__(self):
        self.inverted_index = InvertedIndex()
    
    def add_document(self, doc: Document):
        """Добавление документа в объединенный индекс"""
        self.inverted_index.add_document(doc)
    
    def boolean_search(self, query: str) -> Set[str]:
        """Выполнение булевого поиска с поддержкой полей"""
        from .query import BooleanQueryParser
        parser = BooleanQueryParser(self.inverted_index)
        return parser.parse(query)
    
    def get_available_fields(self) -> Set[str]:
        """Получение списка доступных полей"""
        return self.inverted_index.fields