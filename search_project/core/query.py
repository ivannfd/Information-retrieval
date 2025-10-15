import re
import math
import heapq
from typing import List, Tuple, Dict, Optional, Sequence
from .index import InvertedIndex


class BooleanQueryParser:
    """
    Поддерживаемые конструкции:
      - Бинарные: A OR B, A AND B, left NEAR/n right, left ADJ/n right
      - Унарный: NOT X
      - Скобки: ( ... )
      - Неявный AND: A B  ==>  A AND B
      - Поля: field:term, field:"multi word phrase"
      - Фразы: "foo bar" (позиционно, через координатный индекс)

    Семантика:
      - NEAR/n — неупорядоченная близость: ∃i,j: |pos_i - pos_j| <= n
      - ADJ/n  — упорядоченная близость: ∃i,j: 0 < pos_j - pos_i <= n
                 ADJ без числа трактуется как ADJ/1 (строго соседние).

    """

    _BIN = {"AND", "OR"}
    _UNARY = {"NOT"}

    def __init__(self, index: InvertedIndex):
        self.index = index

    def parse(self, query: str) -> List[str]:
        """Парсит запрос и возвращает ОТСОРТИРОВАННЫЙ список doc_id."""
        tokens = self._tokenize(query)  # токенезация
        tokens = self._normalize_ops(tokens)  # нормализация
        tokens = self._insert_implicit_and(tokens)  # вставляем and между словами
        rpn = self._to_rpn(tokens)  # постфиксная запись
        ast = self._rpn_to_ast(rpn)  # абстрактное синтаксическое дерево
        return self._eval(ast)  # исполнитель

    def _tokenize(self, s: str) -> List[str]:
        """
        Разбивает строку запроса на токены.

        Типы токенов:
          - термы (обычные слова);
          - термы с полем вида `field:term`;
          - фразы в кавычках: `"..."` — сохраняются одним токеном (кавычки остаются);
          - операторы близости: `NEAR/n`, `ADJ/n` — передаются как есть;
          - булевы операторы: `AND`, `OR`, `NOT` (регистр нормализуется позже);
          - скобки: `(` и `)` как отдельные токены.

        Правила:
          - пробел завершает текущий токен;
          - кроме кавычек и скобок символы просто накапливаются в текущий токен;
          - экранирование кавычек не поддерживается: незакрытая кавычка тянет всё до конца;
          - знаки пунктуации не выделяются отдельно.

        Пример:
          'title:"deep learning" AND (graph ADJ/2 neural)'
           ['title:"deep learning"', 'AND', '(', 'graph', 'ADJ/2', 'neural', ')']
        """
        s = s.lower()

        tokens: List[str] = []
        buf: List[str] = []
        i = 0

        def flush():
            if buf:
                tok = "".join(buf).strip()
                if tok:
                    tokens.append(tok)
                buf.clear()

        while i < len(s):

            ch = s[i]

            if ch == '"':
                buf.append(ch)
                i += 1
                while i < len(s):
                    buf.append(s[i])
                    if s[i] == '"':
                        i += 1
                        break
                    i += 1
                flush()
                continue

            if ch in "()":
                flush()
                tokens.append(ch)
                i += 1
                continue

            if ch.isspace():
                flush()
                i += 1
                continue

            buf.append(ch)
            i += 1

        flush()

        return tokens

    def _normalize_ops(self, tokens: List[str]) -> List[str]:
        """
        Нормализует операторные токены в последовательности.

        Делает следующее:
          - булевы операторы приводятся к ВЕРХНЕМУ РЕГИСТРУ: AND / OR / NOT;
          - одиночный `ADJ` (без числа) превращается в `ADJ/1` — строгое соседство;
          - токены вида `NEAR/n` и `ADJ/n` (где n — целое) распознаются и оставляются как есть;
          - все прочие токены (термы, фразы, `field:term`) возвращаются без изменений.
        """
        out = []
        prox_re = r"(NEAR|ADJ)/\d+"
        for t in tokens:
            tu = t.upper()
            if tu == "ADJ":
                out.append("ADJ/1")
            elif tu == "NEAR":
                out.append("NEAR/1")
            elif tu in self._BIN or tu in self._UNARY or re.fullmatch(prox_re, tu):
                out.append(tu)
            else:
                out.append(t)
        return out

    def _insert_implicit_and(self, tokens: List[str]) -> List[str]:
        """Вставляет AND между термами/скобками, где нет явного оператора."""

        out: List[str] = []
        prev: Optional[str] = None

        def is_term(tok: str) -> bool:
            tu = tok.upper()
            return (
                    tok not in ("(", ")")
                    and tu not in self._BIN
                    and tu not in self._UNARY
                    and not re.fullmatch(r"(NEAR|ADJ)/\d+", tu)
            )

        for t in tokens:
            if prev is not None:
                if (is_term(prev) or prev == ")") and (is_term(t) or t == "("):
                    out.append("AND")
            out.append(t)
            prev = t
        return out

    def _to_rpn(self, tokens: List[str]) -> List[str]:
        """Обратная польская запись (постфиксная). Приоритет: OR(1) < AND(2) < NEAR/ADJ(3) < NOT(4)."""
        prec = {"OR": 1, "AND": 2, "NEAR": 3, "ADJ": 3, "NOT": 4}
        right_assoc = {"NOT"}
        out: List[str] = []
        stack: List[str] = []

        def name(op: str) -> str:
            return op.split("/", 1)[0] if "/" in op else op

        for t in tokens:
            tu = t.upper()
            if tu in self._BIN or tu in self._UNARY or re.fullmatch(r"(NEAR|ADJ)/\d+", tu):
                on = name(tu)
                while stack:
                    top = stack[-1]
                    if top == "(":
                        break
                    tn = name(top)
                    if (prec[tn] > prec[on]) or (prec[tn] == prec[on] and on not in right_assoc):
                        out.append(stack.pop());
                        continue
                    break
                stack.append(tu)
            elif t == "(":
                stack.append(t)
            elif t == ")":
                while stack and stack[-1] != "(":
                    out.append(stack.pop())
                if not stack:
                    raise ValueError("Unbalanced parentheses")
                stack.pop()
            else:
                out.append(t)

        while stack:
            top = stack.pop()
            if top in ("(", ")"):
                raise ValueError("Unbalanced parentheses")
            out.append(top)
        return out

    # вводим узлы дерева
    class _Node:
        pass

    class _Term(_Node):
        def __init__(self, token: str) -> None:
            self.token = token

    class _Not(_Node):
        def __init__(self, child: "_Node") -> None:
            self.child = child

    class _Nary(_Node):
        def __init__(self, op: str, children: List["_Node"]) -> None:
            self.op = op
            self.children = children

    class _Prox(_Node):
        def __init__(self, op: str, distance: int, left: "_Node", right: "_Node") -> None:
            self.op = op
            self.distance = distance
            self.left = left
            self.right = right

    def _rpn_to_ast(self, rpn: List[str]) -> "_Node":
        """
            Строит AST (абстрактное синтаксическое дерево) из выражения в обратной польской записи (RPN).

            Вход:
              rpn — список токенов в постфиксной форме, например:
                    ['apple', 'banana', 'AND', 'orange', 'OR']

            Выход:
              Корневой узел дерева (_Node). Листья — _Term; внутренние узлы:
                - _Not(child)           — унарный NOT;
                - _Nary('AND'|'OR', ...)— n-арные AND/OR (внутренние AND/OR сплющиваются);
                - _Prox('NEAR'|'ADJ', n, left, right) — позиционные операторы с дистанцией n.

            Алгоритм:
              Проходим rpn слева направо, ведём стек узлов:
                - терм/фразу → кладём как _Term;
                - NOT → снимаем 1 узел, заворачиваем в _Not, кладём обратно;
                - AND/OR  → снимаем 2 узла (правый, левый), создаём _Nary;
                                если дочерний — такой же _Nary, его дети «сплющиваются»;
                - NEAR/ADJ/n → снимаем 2 узла (правый, левый), создаём _Prox(op, n, left, right).

              В конце в стеке должен остаться ровно один узел — корень дерева.
        """

        st: List[BooleanQueryParser._Node] = []
        for t in rpn:
            tu = t.upper()
            if tu in self._UNARY:
                if not st: raise ValueError("NOT without operand")
                st.append(self._Not(st.pop()))
            elif tu in self._BIN or re.fullmatch(r"(NEAR|ADJ)/\d+", tu):
                if len(st) < 2: raise ValueError(f"{t} requires two operands")
                right, left = st.pop(), st.pop()
                if tu in ("AND", "OR"):
                    node = self._Nary(tu, [])
                    for child in (left, right):
                        if isinstance(child, self._Nary) and child.op == tu:
                            node.children.extend(child.children)
                        else:
                            node.children.append(child)
                    st.append(node)
                else:
                    name, n = tu.split("/", 1)
                    st.append(self._Prox(name, int(n), left, right))
            else:
                st.append(self._Term(t))
        if len(st) != 1:
            raise ValueError("Malformed query")
        return st[0]

    def _eval(self, node: "_Node") -> List[str]:
        if isinstance(node, self._Term):
            field, term = self._split_field(node.token)
            if self._is_quoted(term):
                parts = self._split_phrase_terms(term[1:-1])
                return self._phrase_docids(parts, field)
            postings = self.index.get_postings(term, field=field) or {}
            return sorted(postings.keys())

        if isinstance(node, self._Not):
            universe = self._all_docs_sorted()
            neg = self._eval(node.child)
            return self._diff_sorted(universe, neg)

        if isinstance(node, self._Nary):
            lists = [self._eval(ch) for ch in node.children]
            if node.op == "AND":
                return self._intersect_many_kway(lists)
            else:
                return self._union_many_heap(lists)

        if isinstance(node, self._Prox):
            if not isinstance(node.left, self._Term) or not isinstance(node.right, self._Term):
                raise ValueError(f"{node.op} can be used only with simple terms/phrases (optionally field-scoped).")

            lf, lt = self._split_field(node.left.token)
            rf, rt = self._split_field(node.right.token)

            if lf != rf:
                raise ValueError(f"{node.op} operands must be in the same field (or without field).")

            left_pos = self.index.get_postings(lt, lf) or {}
            right_pos = self.index.get_postings(rt, rf) or {}

            docs = self._intersect_two(sorted(left_pos.keys()), sorted(right_pos.keys()))
            out: List[str] = []

            for doc in docs:
                a = sorted(left_pos[doc])
                b = sorted(right_pos[doc])
                ok = self._within_unordered(a, b, node.distance) if node.op == "NEAR" \
                    else self._within_ordered(a, b, node.distance)
                if ok:
                    out.append(doc)
            return out

        raise AssertionError("Unknown node type")

    def _positions_for_token(self, token: str, field: Optional[str]) -> Dict[str, List[int]]:
        """Терм или фраза -> позиции. Для фразы возвращаем позиции ПОСЛЕДНЕГО слова."""
        if self._is_quoted(token):
            parts = self._split_phrase_terms(token[1:-1])
            return self._phrase_positions(parts, field)
        else:
            return self.index.get_postings(token, field=field) or {}

    def _phrase_docids(self, terms: List[str], field: Optional[str]) -> List[str]:
        pos = self._phrase_positions(terms, field)
        return sorted(pos.keys())

    def _phrase_positions(self, terms: List[str], field: Optional[str]) -> Dict[str, List[int]]:
        """Фразовый поиск: k слов. 1) k-way пересечение docID по словам; 2) позиционная склейка."""
        if not terms:
            return {}
        postings_list: List[Dict[str, List[int]]] = [self.index.get_postings(t, field=field) or {} for t in terms]

        doc_lists = [sorted(p.keys()) for p in postings_list]
        common_docs = self._intersect_many_kway(doc_lists)
        if not common_docs:
            return {}

        result: Dict[str, List[int]] = {}
        for doc in common_docs:
            chains = sorted(postings_list[0][doc])
            for k in range(1, len(terms)):
                nxt = sorted(postings_list[k][doc])
                chains = self._adjacent_positions(chains, nxt)
                if not chains:
                    break
            if chains:
                result[doc] = chains
        return result

    @staticmethod
    def _adjacent_positions(a: List[int], b: List[int]) -> List[int]:
        """Из пар (i∈a, j∈b) оставляет j, где j == i+1. Возвращает отсортированный список j."""
        i = j = 0
        out: List[int] = []
        while i < len(a) and j < len(b):
            diff = b[j] - a[i]
            if diff == 1:
                out.append(b[j])
                i += 1
                j += 1
            elif diff <= 0:
                j += 1
            else:
                i += 1
        return out

    @staticmethod
    def _make_skip_step(n: int) -> int:
        """Длина прыжка по skip-пойнтеру (классика: ⌊√n⌋, но >=2 чтобы имело смысл)."""
        if n <= 3:
            return 0
        return max(2, int(math.sqrt(n)))

    @staticmethod
    def _advance_with_skip(arr: List[str], idx: int, target: str, step: int) -> int:
        """
        Продвигает указатель idx вперёд до первого элемента >= target,
        используя прыжки длиной `step`: пока arr[idx+step] <= target — прыгаем.
        """
        n = len(arr)
        if step <= 0:
            while idx < n and arr[idx] < target:
                idx += 1
            return idx

        while idx + step < n and arr[idx + step] <= target:
            idx += step
        while idx < n and arr[idx] < target:
            idx += 1
        return idx

    def _intersect_many_kway(self, lists: Sequence[List[str]]) -> List[str]:
        """
        K-way пересечение k отсортированных списков docID.
        Алгоритм: поддерживаем указатель по каждому списку; синхронизируемся на max(head);
        короткие списки отрабатывают быстро, длинные используют skip-скачки.
        """
        if any(not lst for lst in lists):
            return []
        if len(lists) == 0:
            return []
        if len(lists) == 1:
            return lists[0]
        lsts = list(lists)
        lsts.sort(key=len)
        k = len(lsts)
        idx = [0] * k
        steps = [self._make_skip_step(len(l)) for l in lsts]
        out: List[str] = []

        def current_heads():
            return [lsts[i][idx[i]] for i in range(k)]

        while True:
            for i in range(k):
                if idx[i] >= len(lsts[i]):
                    return out

            heads = current_heads()
            target = max(heads)

            all_equal = True
            for i in range(k):
                if lsts[i][idx[i]] < target:
                    idx[i] = self._advance_with_skip(lsts[i], idx[i], target, steps[i])
                    if idx[i] >= len(lsts[i]):
                        return out

                if lsts[i][idx[i]] != target:
                    all_equal = False

            if all_equal:
                out.append(target)
                for i in range(k):
                    idx[i] += 1
            else:
                continue

    def _union_many_heap(self, lists: Sequence[List[str]]) -> List[str]:
        """K-way объединение k отсортированных списков docID через кучу, с дедупликацией."""
        lsts = [lst for lst in lists if lst]
        if not lsts:
            return []
        if len(lsts) == 1:
            return lsts[0]

        heap: List[Tuple[str, int]] = []
        iters = [iter(lst) for lst in lsts]
        for idx, it in enumerate(iters):
            try:
                x = next(it)
                heap.append((x, idx))
            except StopIteration:
                pass
        heapq.heapify(heap)

        out: List[str] = []
        last: Optional[str] = None
        while heap:
            doc, idx = heapq.heappop(heap)
            if doc != last:
                out.append(doc);
                last = doc
            try:
                x = next(iters[idx])
                heapq.heappush(heap, (x, idx))
            except StopIteration:
                pass
        return out

    @staticmethod
    def _intersect_two(a: List[str], b: List[str]) -> List[str]:
        """Двухпутевое пересечение без skip (используется точечно)."""
        i = j = 0
        out: List[str] = []
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                out.append(a[i]);
                i += 1;
                j += 1
            elif a[i] < b[j]:
                i += 1
            else:
                j += 1
        return out

    @staticmethod
    def _diff_sorted(universe: List[str], exclude: List[str]) -> List[str]:
        """Разность отсортированных списков: universe \ exclude."""
        i = j = 0
        out: List[str] = []
        while i < len(universe) and j < len(exclude):
            if universe[i] == exclude[j]:
                i += 1;
                j += 1
            elif universe[i] < exclude[j]:
                out.append(universe[i])
                i += 1
            else:
                j += 1
        if i < len(universe):
            out.extend(universe[i:])
        return out

    @staticmethod
    def _within_unordered(a: List[int], b: List[int], n: int) -> bool:
        """NEAR/n: |ai - bj| <= n для некоторых i,j."""
        i = j = 0
        while i < len(a) and j < len(b):
            d = a[i] - b[j]
            if abs(d) <= n:
                return True
            if d < 0:
                i += 1
            else:
                j += 1
        return False

    @staticmethod
    def _within_ordered(a: List[int], b: List[int], n: int) -> bool:
        """ADJ/n: 0 < bj - ai <= n для некоторых i,j (b правее a)."""
        i = j = 0
        while i < len(a) and j < len(b):
            diff = b[j] - a[i]
            if 0 < diff <= n:
                return True
            if diff <= 0:
                j += 1
            else:
                i += 1
        return False

    @staticmethod
    def _is_quoted(term: str) -> bool:
        return len(term) >= 2 and term[0] == '"' and term[-1] == '"'

    @staticmethod
    def _split_phrase_terms(phrase: str) -> List[str]:
        return [t for t in phrase.strip().split() if t]

    @staticmethod
    def _split_field(token: str) -> Tuple[Optional[str], str]:
        if ":" in token and not token.startswith('"'):
            field, term = token.split(":", 1)
            return field.strip(), term.strip()
        return None, token

    def _all_docs_sorted(self) -> List[str]:
        return sorted(self.index.documents.keys())
