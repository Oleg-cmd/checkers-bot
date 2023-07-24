import copy


class CheckersAI:
    def __init__(self, player_color, max_depth=3):
        self.player_color = player_color
        self.max_depth = max_depth

    def evaluate_board(self, board):
        player_score = 0
        opponent_score = 0

        # Весовые коэффициенты
        center_weight = 2
        king_weight = 3
        edge_weight = 1
        attack_weight = 1
        mobility_weight = 0.1
        promote_weight = 3
        king_defense_weight = 1
        king_attack_weight = 1

        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece == self.player_color:
                    player_score += 1
                    if row == 0 or row == 7:
                        # Бонус за дамку, находящуюся близко к превращению
                        player_score += promote_weight
                    if piece == "WK":
                        # Бонус за дамку
                        player_score += king_weight
                        # Бонус за контроль края
                        if col == 0 or col == 7:
                            player_score += edge_weight
                        # Бонус за защиту дамки
                        player_score += self.get_king_defense_score(
                            board, row, col) * king_defense_weight
                        # Бонус за атаку дамки
                        player_score += self.get_king_attack_score(
                            board, row, col) * king_attack_weight
                    else:
                        # Бонус за контроль края
                        if col == 0 or col == 7:
                            player_score += edge_weight
                    # Бонус за атаку
                    player_score += self.get_attack_score(
                        board, row, col) * attack_weight
                    # Бонус за мобильность
                    player_score += len(self.get_valid_moves_for_piece(board,
                                        row, col, piece)) * mobility_weight
                elif piece == self.opponent_color(self.player_color):
                    opponent_score += 1

        return player_score - opponent_score

    def get_king_defense_score(self, board, row, col):
        """
        Вычисляет оценку защиты дамки.
        """
        defense_score = 0
        # Возможные направления защиты дамки
        defense_directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)]
        for direction in defense_directions:
            new_row, new_col = row + direction[0], col + direction[1]
            if 0 <= new_row < 8 and 0 <= new_col < 8 and board[new_row][new_col] == self.player_color:
                defense_score += 1
        return defense_score

    def get_king_attack_score(self, board, row, col):
        """
        Вычисляет оценку атаки дамки.
        """
        attack_score = 0
        # Возможные направления атаки дамки
        attack_directions = [(1, -1), (1, 1), (-1, -1), (-1, 1)]
        for direction in attack_directions:
            new_row, new_col = row + direction[0], col + direction[1]
            if 0 <= new_row < 8 and 0 <= new_col < 8 and board[new_row][new_col] == self.opponent_color(self.player_color):
                attack_score += 1
        return attack_score

    def get_attack_score(self, board, row, col):
        """
        Вычисляет оценку атаки обычной шашки.
        """
        attack_score = 0
        # Возможные направления атаки
        attack_directions = [(1, -1), (1, 1)]
        for direction in attack_directions:
            new_row, new_col = row + direction[0], col + direction[1]
            if 0 <= new_row < 8 and 0 <= new_col < 8 and board[new_row][new_col] == self.opponent_color(self.player_color):
                attack_score += 1
        return attack_score

    @staticmethod
    def opponent_color(color):
        return "B" if color == "W" else "W"

    def king_symbol(self, color):
        return "WK" if color == "W" else "BK"

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

        # Проверка возможности битья назад для дамки
        if board[row][col] == self.king_symbol(color):
            backward_jump_directions = [(-2, -2), (-2, 2)]

            for direction in backward_jump_directions:
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
        piece = new_board[start[0]][start[1]]
        new_board[end[0]][end[1]] = piece
        new_board[start[0]][start[1]] = "_"

        if abs(start[0] - end[0]) == 2:
            # Capture move, remove the captured piece
            middle_row, middle_col = (
                start[0] + end[0]) // 2, (start[1] + end[1]) // 2
            new_board[middle_row][middle_col] = "_"

        # If a piece reaches the last row, promote it to a king
        if end[0] == 0 and new_board[end[0]][end[1]] == "W":
            new_board[end[0]][end[1]] = "WK"
        elif end[0] == 7 and new_board[end[0]][end[1]] == "B":
            new_board[end[0]][end[1]] = "BK"

        return new_board

    # Function to print the graphical representation of the move
    def print_move(self, board, move):
        start, end = move
        piece = board[start[0]][start[1]]
        direction = "↘" if piece == "B" else "↗"
        print(
            f"Move {piece} from ({start[0]}, {start[1]}) {direction} to ({end[0]}, {end[1]})")

    # Function to print the board in a visually appealing way
    def print_board(self, board):
        print("  0 1 2 3 4 5 6 7 ")
        for row in range(8):
            print(row, end=" ")
            for col in range(8):
                if board[row][col] == "_":
                    print("·", end=" ")
                else:
                    print(board[row][col], end=" ")
            print()
        print()

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
        best_capture_count = 0  # Число захваченных фишек лучшего хода
        final_board = None

        possible_moves = self.get_possible_moves(board, self.player_color)

        # Проверяем, есть ли возможность для битья
        capturing_moves = [move for move in possible_moves if abs(
            move[0][0] - move[1][0]) == 2]
        if capturing_moves:
            for move in capturing_moves:
                new_board = self.make_move(copy.deepcopy(board), move)
                capture_count = abs(move[0][0] - move[1][0])
                if capture_count > best_capture_count:
                    best_capture_count = capture_count
                    best_move = move
                    final_board = new_board
        else:
            # Иначе выбираем наилучший ход среди всех возможных ходов
            best_eval = float("-inf")
            for move in possible_moves:
                new_board = self.make_move(copy.deepcopy(board), move)
                eval = self.minimax(new_board, self.max_depth - 1, False)

                if eval > best_eval:
                    best_eval = eval
                    best_move = move
                    final_board = new_board

        self.print_move(board, best_move)
        print()
        print()
        self.print_board(final_board)
        return best_move


# Пример использования:
board = [
    ["_", "W", "_", "W", "_", "W", "_", "W"],
    ["W", "_", "W", "_", "_", "_", "_", "_"],
    ["_", "_", "_", "_", "_", "_", "_", "_"],
    ["_", "_", "_", "_", "W", "_", "W", "_"],
    ["_", "_", "_", "", "_", "", "_", "_"],
    ["B", "_", "B", "_", "B", "_", "B", "_"],
    ["_", "_", "_", "_", "_", "_", "_", "B"],
    ["_", "_", "B", "_", "B", "_", "B", "_"],
]

ai = CheckersAI("B", max_depth=5)
best_move = ai.get_best_move(board)
