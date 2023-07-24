import copy


class CheckersAI:
    def __init__(self, player_color, max_depth=3):
        self.player_color = player_color
        self.max_depth = max_depth

    def evaluate_board(self, board):
        # Простая эвристика для оценки доски. Можно улучшить, чтобы получить более сильного ИИ.
        player_pieces = 0
        opponent_pieces = 0
        for row in board:
            player_pieces += row.count(self.player_color)
            opponent_pieces += row.count(
                self.opponent_color(self.player_color))
        return player_pieces - opponent_pieces

    @staticmethod
    def opponent_color(color):
        return "B" if color == "W" else "W"

    def get_possible_moves(self, board, color):
        moves = []
        for r in range(8):
            for c in range(8):
                if board[r][c] == color:
                    moves.extend(self.get_valid_moves_for_piece(
                        board, r, c, color))
        return moves

    def get_valid_moves_for_piece(self, board, row, col, color):
        moves = []

        # Проверка возможности движения вперед
        forward_direction = 1 if color == "W" else -1

        # Возможные направления хода для шашки
        move_directions = [(forward_direction, -1), (forward_direction, 1)]

        for direction in move_directions:
            new_row, new_col = row + direction[0], col + direction[1]
            if 0 <= new_row < 8 and 0 <= new_col < 8 and board[new_row][new_col] == "_":
                moves.append(((row, col), (new_row, new_col)))

        # Проверка возможности битья
        jump_directions = [(2 * forward_direction, -2),
                           (2 * forward_direction, 2)]

        for direction in jump_directions:
            new_row, new_col = row + direction[0], col + direction[1]
            enemy_row, enemy_col = row + \
                direction[0] // 2, col + direction[1] // 2

            if (
                0 <= new_row < 8
                and 0 <= new_col < 8
                and board[new_row][new_col] == "_"
                and 0 <= enemy_row < 8
                and 0 <= enemy_col < 8
                and board[enemy_row][enemy_col] == self.opponent_color(color)
            ):
                moves.append(((row, col), (new_row, new_col)))

        return moves

    def make_move(self, board, move):
        new_board = copy.deepcopy(board)
        start, end = move
        new_board[end[0]][end[1]] = new_board[start[0]][start[1]]
        new_board[start[0]][start[1]] = "_"

        if abs(start[0] - end[0]) == 2:
            # Выполняем взятие, удаляем побитую шашку
            middle_row, middle_col = (
                start[0] + end[0]) // 2, (start[1] + end[1]) // 2
            new_board[middle_row][middle_col] = "_"

        # Если шашка добралась до последнего ряда, превращаем ее в дамку
        if end[0] == 0 and new_board[end[0]][end[1]] == "W":
            new_board[end[0]][end[1]] = "WK"
        elif end[0] == 7 and new_board[end[0]][end[1]] == "B":
            new_board[end[0]][end[1]] = "BK"

        return new_board

    def opposite_color(color):
        return "B" if color == "W" else "W"

    def minimax(self, board, depth, maximizing_player):
        if depth == 0:
            return self.evaluate_board(board)

        сolor = self.player_color if maximizing_player else CheckersAI.opponent_color(
            self.player_color)
        possible_moves = self.get_possible_moves(board, сolor)

        if maximizing_player:
            max_eval = float("-inf")
            for move in possible_moves:
                new_board = self.make_move(copy.deepcopy(board), move)
                eval = self.minimax(new_board, depth - 1, False)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float("inf")
            for move in possible_moves:
                new_board = self.make_move(copy.deepcopy(board), move)
                eval = self.minimax(new_board, depth - 1, True)
                min_eval = min(min_eval, eval)
            return min_eval

    def get_best_move(self, board):
        best_move = None
        best_eval = float("-inf")

        possible_moves = self.get_possible_moves(board, self.player_color)

        for move in possible_moves:
            new_board = self.make_move(copy.deepcopy(board), move)
            eval = self.minimax(new_board, self.max_depth - 1, False)

            if eval > best_eval:
                best_eval = eval
                best_move = move

        return best_move


# Пример использования:
board = [
    ["_", "W", "_", "W", "_", "W", "_", "W"],
    ["W", "_", "W", "_", "W", "_", "W", "_"],
    ["_", "W", "_", "W", "_", "W", "_", "W"],
    ["_", "_", "_", "_", "_", "_", "_", "_"],
    ["_", "_", "_", "_", "_", "_", "_", "_"],
    ["B", "_", "B", "_", "B", "_", "B", "_"],
    ["_", "B", "_", "B", "_", "B", "_", "B"],
    ["B", "_", "B", "_", "B", "_", "B", "_"],
]

ai = CheckersAI("B", max_depth=5)
best_move = ai.get_best_move(board)
print(f"The best move is: {best_move}")
