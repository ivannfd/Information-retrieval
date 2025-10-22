import math
import struct
import time
from collections import defaultdict
from typing import DefaultDict
from typing import List, Tuple, Dict, Set

from document import Document
from tokenizer import Tokenizer


class InvertedIndex:
    def __init__(self):
        # стуктура индекса: term -> field -> doc_id -> positions
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


class PForDeltaCompressor:
    """
    PForDelta компрессор
    """

    def __init__(self, block_size: int = 128, exception_threshold: float = 0.1):
        self.block_size = block_size
        self.exception_threshold = exception_threshold

    def compress_doc_ids(self, doc_ids: List[int]) -> bytes:
        """
        Сжатие списка doc IDs с использованием PForDelta.
        
        Args:
            doc_ids: Отсортированный список целочисленных ID документов
            
        Returns:
            Сжатые данные в виде bytes
        """
        if not doc_ids:
            return b''

        sorted_ids = sorted(doc_ids)

        deltas = self._to_deltas(sorted_ids)

        blocks = [deltas[i:i + self.block_size]
                  for i in range(0, len(deltas), self.block_size)]

        compressed_blocks = []
        for block in blocks:
            compressed_block = self._compress_block(block)
            compressed_blocks.append(compressed_block)

        return self._pack_blocks(compressed_blocks, len(doc_ids))

    def decompress_doc_ids(self, data: bytes) -> List[int]:
        """
        Декомпрессия списка doc IDs.
        """
        if not data:
            return []

        blocks, original_length = self._unpack_blocks(data)

        decompressed_deltas = []
        for block_data in blocks:
            block_deltas = self._decompress_block(block_data)
            decompressed_deltas.extend(block_deltas)

        return self._from_deltas(decompressed_deltas)

    def _compress_block(self, block: List[int]) -> Tuple[int, List[int], List[int]]:
        """Сжатие одного блока дельт."""
        if not block:
            return 0, [], []

        b = self._find_optimal_b(block)

        compressed_data = []
        exceptions = []

        mask = (1 << b) - 1

        for num in block:
            if num <= mask:
                compressed_data.append(num)
            else:
                compressed_data.append(mask)
                exceptions.append(num)

        return b, compressed_data, exceptions

    def _find_optimal_b(self, block: List[int]) -> int:
        """Находит оптимальное количество бит для кодирования блока."""
        if not block:
            return 0

        max_val = max(block)
        if max_val == 0:
            return 1

        for b in range(1, 32):
            mask = (1 << b) - 1
            exceptions = sum(1 for num in block if num > mask)

            if exceptions <= len(block) * self.exception_threshold:
                return b

        return math.ceil(math.log2(max_val + 1))

    def _decompress_block(self, block_data: Tuple[int, List[int], List[int]]) -> List[int]:
        """Декомпрессия одного блока."""
        b, compressed_data, exceptions = block_data

        if b == 0:
            return []

        result = []
        exception_index = 0
        mask = (1 << b) - 1

        for num in compressed_data:
            if num == mask and exception_index < len(exceptions):
                result.append(exceptions[exception_index])
                exception_index += 1
            else:
                result.append(num)

        return result

    def _to_deltas(self, numbers: List[int]) -> List[int]:
        """Конвертирует отсортированный список в дельты."""
        if not numbers:
            return []

        deltas = [numbers[0]]
        for i in range(1, len(numbers)):
            delta = numbers[i] - numbers[i - 1]
            deltas.append(delta)

        return deltas

    def _from_deltas(self, deltas: List[int]) -> List[int]:
        """Восстанавливает отсортированный список из дельт."""
        if not deltas:
            return []

        numbers = [deltas[0]]
        for i in range(1, len(deltas)):
            numbers.append(numbers[-1] + deltas[i])

        return numbers

    def _pack_blocks(self, blocks: List[Tuple[int, List[int], List[int]]],
                     original_length: int) -> bytes:
        """Упаковывает блоки в байтовую строку."""
        result = bytearray()

        result.extend(struct.pack('<II', original_length, len(blocks)))

        for b, compressed_data, exceptions in blocks:
            result.extend(struct.pack('<BII', b, len(compressed_data), len(exceptions)))

            if b > 0 and compressed_data:
                packed_data = self._pack_bit_array(compressed_data, b)
                result.extend(struct.pack('<I', len(packed_data)))
                result.extend(packed_data)
            else:
                result.extend(struct.pack('<I', 0))

            if exceptions:
                packed_exceptions = struct.pack('<' + 'I' * len(exceptions), *exceptions)
                result.extend(packed_exceptions)

        return bytes(result)

    def _unpack_blocks(self, data: bytes) -> Tuple[List[Tuple[int, List[int], List[int]]], int]:
        """Распаковывает байтовую строку обратно в блоки."""
        if len(data) < 8:
            return [], 0

        data_view = memoryview(data)
        offset = 0

        original_length, num_blocks = struct.unpack_from('<II', data_view, offset)
        offset += 8

        blocks = []

        for _ in range(num_blocks):
            if offset + 9 > len(data):
                break

            b, data_size, exceptions_size = struct.unpack_from('<BII', data_view, offset)
            offset += 9

            compressed_data = []
            exceptions = []

            if b > 0 and data_size > 0:
                if offset + 4 > len(data):
                    break

                packed_size, = struct.unpack_from('<I', data_view, offset)
                offset += 4

                if packed_size > 0 and offset + packed_size <= len(data):
                    packed_data = data_view[offset:offset + packed_size]
                    offset += packed_size

                    compressed_data = self._unpack_bit_array(packed_data.tobytes(), b, data_size)

            if exceptions_size > 0:
                if offset + exceptions_size * 4 <= len(data):
                    exceptions = list(struct.unpack_from('<' + 'I' * exceptions_size, data_view, offset))
                    offset += exceptions_size * 4

            blocks.append((b, compressed_data, exceptions))

        return blocks, original_length

    def _pack_bit_array(self, numbers: List[int], bits_per_number: int) -> bytes:
        """Упаковывает массив чисел в битовый массив."""
        if bits_per_number == 0 or not numbers:
            return b''

        total_bits = len(numbers) * bits_per_number
        total_bytes = (total_bits + 7) // 8

        result = bytearray(total_bytes)

        for i, num in enumerate(numbers):
            bit_offset = i * bits_per_number
            byte_index = bit_offset // 8
            bit_index = bit_offset % 8

            for bit in range(bits_per_number):
                if byte_index >= len(result):
                    break

                bit_value = (num >> (bits_per_number - 1 - bit)) & 1
                result[byte_index] |= (bit_value << (7 - bit_index))

                bit_index += 1
                if bit_index >= 8:
                    byte_index += 1
                    bit_index = 0

        return bytes(result)

    def _unpack_bit_array(self, data: bytes, bits_per_number: int, count: int) -> List[int]:
        """Распаковывает битовый массив обратно в числа."""
        if bits_per_number == 0 or count == 0 or not data:
            return []

        result = []
        data_view = memoryview(data)

        for i in range(count):
            bit_offset = i * bits_per_number
            byte_index = bit_offset // 8
            bit_index = bit_offset % 8

            value = 0

            for bit in range(bits_per_number):
                if byte_index >= len(data_view):
                    break

                bit_value = (data_view[byte_index] >> (7 - bit_index)) & 1
                value = (value << 1) | bit_value

                bit_index += 1
                if bit_index >= 8:
                    byte_index += 1
                    bit_index = 0

            result.append(value)

        return result


class SimplePositionStorage:
    """
    Простое хранилище позиций. Позиции хранятся без сложного сжатия,
    так как они обычно короткие и PForDelta для них неэффективен.
    """

    def compress_positions(self, positions: List[int]) -> bytes:
        """Простое кодирование позиций."""
        if not positions:
            return b''

        positions.sort()

        result = bytearray()

        if len(positions) < 128:
            result.append(len(positions))
        else:
            result.append(0x80 | (len(positions) >> 8))
            result.append(len(positions) & 0xFF)

        last_pos = 0
        for pos in positions:
            delta = pos - last_pos
            last_pos = pos

            while delta >= 128:
                result.append((delta & 0x7F) | 0x80)
                delta >>= 7
            result.append(delta)

        return bytes(result)

    def decompress_positions(self, data: bytes) -> List[int]:
        """Декомпрессия позиций."""
        if not data:
            return []

        result = []
        offset = 0

        if data[0] & 0x80:
            count = ((data[0] & 0x7F) << 8) | data[1]
            offset = 2
        else:
            count = data[0]
            offset = 1

        last_pos = 0
        for _ in range(count):
            if offset >= len(data):
                break

            delta = 0
            shift = 0
            while offset < len(data):
                byte_val = data[offset]
                offset += 1
                delta |= (byte_val & 0x7F) << shift
                if not (byte_val & 0x80):
                    break
                shift += 7

            last_pos += delta
            result.append(last_pos)

        return result


class CompressedInvertedIndex(InvertedIndex):
    """
    Реализация инвертированного индекса с PForDelta сжатием posting lists.
    """

    def __init__(self, compression_block_size: int = 128, batch_size=50000):
        super().__init__()
        self.doc_compressor = PForDeltaCompressor(block_size=compression_block_size)
        self.pos_storage = SimplePositionStorage()

        self.batch_size = batch_size

        self.doc_index = defaultdict(lambda: defaultdict(bytes))
        self.pos_index = defaultdict(lambda: defaultdict(lambda: defaultdict(bytes)))

        self.doc_id_to_int = {}
        self.int_to_doc_id = {}
        self.next_doc_id = 1

        self._pending_docs = []
        self._pending_positions = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def _get_numeric_doc_id(self, doc_id: str) -> int:
        """Получает числовой ID для строкового doc_id."""
        if doc_id not in self.doc_id_to_int:
            self.doc_id_to_int[doc_id] = self.next_doc_id
            self.int_to_doc_id[self.next_doc_id] = doc_id
            self.next_doc_id += 1
        return self.doc_id_to_int[doc_id]

    def _get_string_doc_id(self, numeric_id: int) -> str:
        """Получает строковый ID для числового doc_id."""
        return self.int_to_doc_id.get(numeric_id, "")

    def add_document(self, doc: Document):
        """Добавление документа в батч-буфер"""
        self.documents[doc.id] = doc
        numeric_id = self._get_numeric_doc_id(doc.id)

        all_tokens = []
        current_position = 0

        for field_name, field_text in doc.fields.items():
            self.fields.add(field_name)
            tokens_with_positions = self.tokenizer.tokenize_field(field_name, field_text)

            for token, position in tokens_with_positions:
                all_tokens.append((token, field_name, current_position))
                current_position += 1

        self._pending_docs.append((numeric_id, all_tokens))

        if len(self._pending_docs) >= self.batch_size:
            self._process_batch()

    def _process_batch(self):
        """Обработка накопленного батча"""
        if not self._pending_docs:
            return

        start_time = time.time()

        doc_updates = defaultdict(lambda: defaultdict(list))
        pos_updates = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for numeric_id, tokens in self._pending_docs:
            doc_id = self._get_string_doc_id(numeric_id)

            for token, field_name, position in tokens:
                doc_updates[token][field_name].append(numeric_id)
                pos_updates[token][field_name][doc_id].append(position)

        for token, field_data in doc_updates.items():
            for field_name, new_ids in field_data.items():
                self._update_posting_list_batch(token, field_name, new_ids)

        for token, field_data in pos_updates.items():
            for field_name, doc_data in field_data.items():
                for doc_id, positions in doc_data.items():
                    compressed_pos = self.pos_storage.compress_positions(positions)
                    self.pos_index[token][field_name][doc_id] = compressed_pos


        self._pending_docs.clear()

        processing_time = time.time() - start_time
        # print(f"Processed batch of {len(doc_updates)} documents in {processing_time:.3f}s")

    def _update_posting_list_batch(self, token: str, field: str, new_ids: List[int]):
        """Батч-обновление posting list"""

        current_data = self.doc_index[token][field]
        if current_data:
            current_ids = set(self.doc_compressor.decompress_doc_ids(current_data))
        else:
            current_ids = set()

        current_ids.update(new_ids)

        sorted_ids = sorted(current_ids)
        compressed = self.doc_compressor.compress_doc_ids(sorted_ids)
        self.doc_index[token][field] = compressed

    def flush(self):
        """Принудительная обработка оставшихся документов в батче"""
        if self._pending_docs:
            self._process_batch()

    def _get_numeric_doc_id(self, doc_id: str) -> int:
        if doc_id not in self.doc_id_to_int:
            self.doc_id_to_int[doc_id] = self.next_doc_id
            self.int_to_doc_id[self.next_doc_id] = doc_id
            self.next_doc_id += 1
        return self.doc_id_to_int[doc_id]

    def _get_string_doc_id(self, numeric_id: int) -> str:
        return self.int_to_doc_id.get(numeric_id, "")

    def get_postings(self, term: str, field: str = None) -> Dict[str, List[int]]:
        """Получение постингов с декомпрессией"""
        self.flush()
        term = term.lower()
        result = {}

        numeric_doc_ids = self._get_numeric_doc_ids(term, field)

        for numeric_id in numeric_doc_ids:
            doc_id = self._get_string_doc_id(numeric_id)
            positions = self._get_document_positions(term, field, doc_id)
            if positions:
                result[doc_id] = positions

        return result

    def _get_numeric_doc_ids(self, term: str, field: str = None) -> List[int]:
        """Получает числовые doc_ids для термина."""
        term = term.lower()

        if field:
            compressed_data = self.doc_index.get(term, {}).get(field)
            if compressed_data:
                return self.doc_compressor.decompress_doc_ids(compressed_data)
            return []
        else:
            all_numeric_ids = set()
            for field_name in self.doc_index.get(term, {}):
                compressed_data = self.doc_index[term][field_name]
                if compressed_data:
                    field_ids = self.doc_compressor.decompress_doc_ids(compressed_data)
                    all_numeric_ids.update(field_ids)
            return sorted(all_numeric_ids)

    def _get_document_positions(self, term: str, field: str, doc_id: str) -> List[int]:
        """Получает позиции для конкретного документа и термина."""
        term = term.lower()

        if field:
            compressed_pos = self.pos_index.get(term, {}).get(field, {}).get(doc_id)
            if compressed_pos:
                return self.pos_storage.decompress_positions(compressed_pos)
            return []
        else:
            all_positions = []
            for field_name in self.pos_index.get(term, {}):
                compressed_pos = self.pos_index[term][field_name].get(doc_id)
                if compressed_pos:
                    positions = self.pos_storage.decompress_positions(compressed_pos)
                    all_positions.extend(positions)
            return sorted(all_positions)

    def search_term(self, term: str, field: str = None) -> Set[str]:
        """Быстрый поиск doc_ids без декомпрессии позиций"""
        self.flush()
        numeric_doc_ids = self._get_numeric_doc_ids(term, field)
        return {self._get_string_doc_id(numeric_id) for numeric_id in numeric_doc_ids}

    def get_document_positions(self, doc_id: str, field: str, term: str) -> List[int]:
        """Получение позиций термина в конкретном поле документа"""
        return self._get_document_positions(term, field, doc_id)


class SearchEngine:
    def __init__(self, index_type="compressed"):
        if index_type == "compressed":
            self.inverted_index = CompressedInvertedIndex()
        else:
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
