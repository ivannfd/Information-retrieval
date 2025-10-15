from flask import Flask, render_template, request, jsonify
import sys
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.index import SearchEngine
from core.document import Document

app = Flask(__name__)
search_engine = SearchEngine()

cranfield_documents = []
cranfield_queries = []
cranfield_relevance = []


@dataclass
class CranfieldDocument:
    id: str
    title: str
    authors: str
    bibliography: str
    text: str


@dataclass
class CranfieldQuery:
    id: str
    text: str


@dataclass
class RelevanceJudgment:
    query_id: str
    doc_id: str
    relevance: int


class CranfieldParser:
    def __init__(self):
        self.doc_pattern = re.compile(
            r'\.I\s+(\d+)\s*\.T\s*(.*?)\s*\.A\s*(.*?)\s*\.B\s*(.*?)\s*\.W\s*(.*?)(?=\s*\.I\s+\d+|$)', re.DOTALL)
        self.query_pattern = re.compile(r'\.I\s+(\d+)\s*\.W\s*(.*?)(?=\s*\.I\s+\d+|$)', re.DOTALL)

    def parse_documents(self, file_path: str) -> List[CranfieldDocument]:
        """Парсинг файла cran.all.1400"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            documents = []
            matches = self.doc_pattern.findall(content)

            for match in matches:
                doc_id, title, authors, bibliography, text = match
                document = CranfieldDocument(
                    id=f"cran_{doc_id.strip()}",
                    title=self._clean_text(title),
                    authors=self._clean_text(authors),
                    bibliography=self._clean_text(bibliography),
                    text=self._clean_text(text)
                )
                documents.append(document)

            return documents
        except Exception as e:
            print(f"Error parsing documents: {e}")
            return []

    def parse_queries(self, file_path: str) -> List[CranfieldQuery]:
        """Парсинг файла cran.qry"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            queries = []
            matches = self.query_pattern.findall(content)

            for match in matches:
                query_id, query_text = match
                query = CranfieldQuery(
                    id=f"query_{query_id.strip()}",
                    text=self._clean_text(query_text)
                )
                queries.append(query)

            return queries
        except Exception as e:
            print(f"Error parsing queries: {e}")
            return []

    def parse_relevance(self, file_path: str) -> List[RelevanceJudgment]:
        """Парсинг файла cranqrel"""
        try:
            judgments = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        query_id, doc_id = parts[0], parts[1]
                        relevance = int(parts[2]) if len(parts) > 2 else 1
                        judgments.append(RelevanceJudgment(
                            query_id=f"query_{query_id}",
                            doc_id=f"cran_{doc_id}",
                            relevance=relevance
                        ))
            return judgments
        except Exception as e:
            print(f"Error parsing relevance: {e}")
            return []

    def _clean_text(self, text: str) -> str:
        """Очистка текста от лишних пробелов и переносов"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


def initialize_sample_data():
    """Инициализация тестовыми данными"""
    sample_docs = [
        Document.create(
            title="Python Programming",
            content="Python is a great programming language for web development and data science.",
            author="John Doe"
        ),
        Document.create(
            title="Web Development",
            content="Modern web development involves Python, JavaScript, and other technologies.",
            author="Jane Smith"
        ),
        Document.create(
            title="Data Science",
            content="Data science uses Python for machine learning and data analysis.",
            author="John Doe"
        ),
        Document.create(
            title="Machine Learning",
            content="Python is popular for machine learning and artificial intelligence projects.",
            author="Bob Wilson"
        )
    ]

    for doc in sample_docs:
        search_engine.add_document(doc)


def create_search_documents(cranfield_docs: List[CranfieldDocument]) -> List[Document]:
    """Конвертация Cranfield документов в формат для SearchEngine"""
    search_docs = []

    for cran_doc in cranfield_docs:
        doc = Document(
            id=cran_doc.id,
            fields={
                "title": cran_doc.title,
                "authors": cran_doc.authors,
                "bibliography": cran_doc.bibliography,
                "text": cran_doc.text,
                "content": f"{cran_doc.title} {cran_doc.authors} {cran_doc.text}"
            }
        )
        search_docs.append(doc)

    return search_docs


def load_cranfield_dataset(base_path: str = "../cranfield"):
    """Загрузка датасета Cranfield"""
    global cranfield_documents, cranfield_queries, cranfield_relevance

    docs_file = os.path.join(base_path, "cran.all.1400")
    queries_file = os.path.join(base_path, "cran.qry")
    relevance_file = os.path.join(base_path, "cranqrel")

    parser = CranfieldParser()

    cranfield_documents = parser.parse_documents(docs_file)
    cranfield_queries = parser.parse_queries(queries_file)
    cranfield_relevance = parser.parse_relevance(relevance_file)

    return len(cranfield_documents), len(cranfield_queries), len(cranfield_relevance)


def add_cranfield_to_index(doc_count: Optional[int] = None):
    """Добавление документов Cranfield в поисковый индекс"""
    if not cranfield_documents:
        return 0

    docs_to_add = cranfield_documents[:doc_count] if doc_count else cranfield_documents
    search_docs = create_search_documents(docs_to_add)

    for doc in search_docs:
        search_engine.add_document(doc)

    return len(search_docs)


def evaluate_query(query_id: str, top_k: int = 10) -> Dict:
    """Оценка одного запроса"""
    global cranfield_relevance

    # Находим запрос
    query = next((q for q in cranfield_queries if q.id == query_id), None)
    if not query:
        return {"error": "Query not found"}

    # Находим релевантные документы для этого запроса
    relevant_docs = [j.doc_id for j in cranfield_relevance if j.query_id == query_id]

    # Выполняем поиск
    try:
        results = search_engine.boolean_search(query.text)
        retrieved_docs = list(results)[:top_k]
        print(retrieved_docs)
        # Вычисляем метрики
        relevant_retrieved = set(retrieved_docs) & set(relevant_docs)

        precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
        recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "query_id": query_id,
            "query_text": query.text,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "retrieved_count": len(retrieved_docs),
            "relevant_count": len(relevant_docs),
            "relevant_retrieved_count": len(relevant_retrieved),
            "retrieved_docs": retrieved_docs,
            "relevant_docs": relevant_docs
        }
    except Exception as e:
        return {"error": str(e)}


initialize_sample_data()


@app.route('/')
def index():
    fields = list(search_engine.get_available_fields())
    doc_count = len(search_engine.inverted_index.documents)
    return render_template('index.html', fields=fields, doc_count=doc_count)


@app.route('/search')
def search():
    query = request.args.get('q', '')
    if not query:
        return render_template('results.html', query=query, results=[])

    try:
        doc_ids = search_engine.boolean_search(query)
        results = []
        for doc_id in doc_ids:
            doc = search_engine.inverted_index.documents.get(doc_id)
            if doc:
                results.append({
                    'id': doc.id,
                    'title': doc.fields.get('title', 'No title'),
                    'content': doc.fields.get('content', ''),
                    'author': doc.fields.get('author', 'Unknown'),
                    'preview': doc.fields.get('content', '')[:200] + '...' if len(
                        doc.fields.get('content', '')) > 200 else doc.fields.get('content', '')
                })

        return render_template('results.html', query=query, results=results)

    except Exception as e:
        return render_template('results.html', query=query, error=str(e), results=[])


@app.route('/dataset')
def dataset():
    """Страница управления датасетом"""
    doc_count = len(search_engine.inverted_index.documents)
    cranfield_stats = {
        'documents_loaded': len(cranfield_documents),
        'queries_loaded': len(cranfield_queries),
        'relevance_loaded': len(cranfield_relevance)
    }
    return render_template('dataset.html',
                           doc_count=doc_count,
                           cranfield_stats=cranfield_stats)


@app.route('/api/load_cranfield', methods=['POST'])
def api_load_cranfield():
    """API для загрузки датасета Cranfield"""
    # data = request.json
    base_path = '../cranfield'  # data.get('base_path', '../cranfield')

    try:
        docs_count, queries_count, relevance_count = load_cranfield_dataset(base_path)
        return jsonify({
            'success': True,
            'message': f'Loaded {docs_count} documents, {queries_count} queries, {relevance_count} relevance judgments',
            'stats': {
                'documents': docs_count,
                'queries': queries_count,
                'relevance': relevance_count
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/add_cranfield_docs', methods=['POST'])
def api_add_cranfield_docs():
    """API для добавления документов Cranfield в индекс"""
    data = request.json
    doc_count = data.get('doc_count')

    try:
        added_count = add_cranfield_to_index(doc_count)
        total_docs = len(search_engine.inverted_index.documents)
        return jsonify({
            'success': True,
            'message': f'Added {added_count} documents to index',
            'stats': {
                'added': added_count,
                'total': total_docs
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/clear_index', methods=['POST'])
def api_clear_index():
    """API для очистки индекса"""
    try:
        search_engine = SearchEngine()
        initialize_sample_data()

        globals()['search_engine'] = search_engine

        return jsonify({
            'success': True,
            'message': 'Index cleared successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/evaluate_query', methods=['POST'])
def api_evaluate_query():
    """API для оценки запроса"""
    data = request.json
    query_id = data.get('query_id')

    try:
        result = evaluate_query(query_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e)
        })


@app.route('/api/get_queries')
def api_get_queries():
    """API для получения списка запросов"""
    queries_data = [{'id': q.id, 'text': q.text} for q in cranfield_queries[:50]]  # Ограничиваем для производительности
    return jsonify(queries_data)


@app.route('/api/get_dataset_stats')
def api_get_dataset_stats():
    """API для получения статистики датасета"""
    stats = {
        'total_documents': len(search_engine.inverted_index.documents),
        'cranfield_documents': len(cranfield_documents),
        'cranfield_queries': len(cranfield_queries),
        'cranfield_relevance': len(cranfield_relevance),
        'indexed_cranfield_docs': len(
            [d for d in search_engine.inverted_index.documents.values() if d.id.startswith('cran_')])
    }
    return jsonify(stats)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
