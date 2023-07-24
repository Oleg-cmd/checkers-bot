import copy
import pygame
import random


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
        king_defense_weight = 2
        king_attack_weight = 2

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
                        player_score += (
                            self.get_king_defense_score(board, row, col)
                            * king_defense_weight
                        )
                        # Бонус за атаку дамки
                        player_score += (
                            self.get_king_attack_score(board, row, col)
                            * king_attack_weight
                        )
                    else:
                        # Бонус за контроль края
                        if col == 0 or col == 7:
                            player_score += edge_weight
                    # Бонус за атаку
                    player_score += (
                        self.get_attack_score(board, row, col) * attack_weight
                    )
                    # Бонус за мобильность
                    player_score += (
                        len(self.get_valid_moves_for_piece(board, row, col, piece))
                        * mobility_weight
                    )
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
            if (
                0 <= new_row < 8
                and 0 <= new_col < 8
                and board[new_row][new_col] == self.player_color
            ):
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
            if (
                0 <= new_row < 8
                and 0 <= new_col < 8
                and board[new_row][new_col] == self.opponent_color(self.player_color)
            ):
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
            if (
                0 <= new_row < 8
                and 0 <= new_col < 8
                and board[new_row][new_col] == self.opponent_color(self.player_color)
            ):
                attack_score += 1
        return attack_score

    @staticmethod
    def opponent_color(color):
        return "B" if color == "W" else "W"

    def king_symbol(self, color):
        return "WK" if color == "W" else "BK"

    def get_possible_moves(self, board, color, move=None, captured=None):
        moves = []

        for r in range(8):
            for c in range(8):
                if board[r][c] == color:
                    moves.extend(self.get_valid_moves_for_piece(board, r, c, color))

        # Проверим, есть ли обязательные удары
        mandatory_jumps = [move for move in moves if abs(move[0][0] - move[1][0]) == 2]

        # Если обязательные удары есть, вернем только их
        if mandatory_jumps:
            return mandatory_jumps
        else:
            return moves

    def get_valid_moves_for_piece(self, board, row, col, color):
        moves = []

        # Возможные направления для дамки
        king_directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        # Проверка возможности движения вперед
        forward_direction = 1 if color == "W" else -1
        backward_direction = -forward_direction  # Направление для обратного движения

        # Возможные направления хода для шашки
        move_directions = [(forward_direction, -1), (forward_direction, 1)]  # Назад
        fight = []

        # Проверка возможности битья
        jump_directions = [
            (2 * forward_direction, -2),
            (2 * forward_direction, 2),
            (2 * backward_direction, -2),
            (2 * backward_direction, 2),
        ]

        for direction in jump_directions:
            new_row, new_col = row + direction[0], col + direction[1]
            enemy_row, enemy_col = row + direction[0] // 2, col + direction[1] // 2

            if (
                0 <= new_row < 8
                and 0 <= new_col < 8
                and board[new_row][new_col] == "_"
                and 0 <= enemy_row < 8
                and 0 <= enemy_col < 8
                and board[enemy_row][enemy_col] == self.opponent_color(color)
            ):
                fight.append(((row, col), (new_row, new_col)))

        if len(fight) > 0:
            return fight

        # Если нет обязательных ударов, добавляем обычные ходы
        for direction in move_directions:
            new_row, new_col = row + direction[0], col + direction[1]
            if 0 <= new_row < 8 and 0 <= new_col < 8 and board[new_row][new_col] == "_":
                moves.append(((row, col), (new_row, new_col)))

        return moves

    def promote_to_king(self, board, row, col):
        piece = board[row][col]
        if piece == "W" and row == 0:
            board[row][col] = "WK"
        elif piece == "B" and row == 7:
            board[row][col] = "BK"
        return board

    def make_move(self, board, move):
        new_board = copy.deepcopy(board)
        start, end = move
        piece = new_board[start[0]][start[1]]
        new_board[end[0]][end[1]] = piece
        new_board[start[0]][start[1]] = "_"

        if abs(start[0] - end[0]) == 2:
            # Capture move, remove the captured piece
            middle_row, middle_col = (start[0] + end[0]) // 2, (start[1] + end[1]) // 2
            new_board[middle_row][middle_col] = "_"

        new_board = self.promote_to_king(new_board, end[0], end[1])

        return new_board

    # Функция, которая возвращает True, если у шашки есть возможность дополнительных ударов
    def has_extra_jumps(self, board, row, col, color):
        extra_moves = self.get_valid_moves_for_piece(board, row, col, color)
        mandatory_jumps = [
            move for move in extra_moves if abs(move[0][0] - move[1][0]) == 2
        ]
        return bool(mandatory_jumps)

    # Function to print the graphical representation of the move
    def print_move(self, board, move):
        start, end = move
        piece = board[start[0]][start[1]]
        direction = "↘" if piece == "B" else "↗"
        print(
            f"Move {piece} from ({start[0]}, {start[1]}) {direction} to ({end[0]}, {end[1]})"
        )

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

    # default minmax with alfa-beta split
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0:
            return self.evaluate_board(board)

        color = (
            self.player_color
            if maximizing_player
            else CheckersAI.opponent_color(self.player_color)
        )
        possible_moves = self.get_possible_moves(board, color)

        if maximizing_player:
            max_eval = float("-inf")
            for move in possible_moves:
                new_board = self.make_move(copy.deepcopy(board), move)
                eval = self.minimax(new_board, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Альфа-бета отсечение
            return max_eval
        else:
            min_eval = float("inf")
            for move in possible_moves:
                new_board = self.make_move(copy.deepcopy(board), move)
                eval = self.minimax(new_board, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Альфа-бета отсечение
            return min_eval

    def get_best_move(self, board):
        best_moves = []
        best_score = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        possible_moves = self.get_possible_moves(board, self.player_color)

        if possible_moves:
            for move in possible_moves:
                new_board = self.make_move(copy.deepcopy(board), move)
                eval_score = self.minimax(
                    new_board, self.max_depth - 1, alpha, beta, False
                )

                if eval_score > best_score:
                    best_score = eval_score
                    best_moves = [move]
                elif eval_score == best_score:
                    best_moves.append(move)

        if best_moves:
            return best_moves
        else:
            return None


# drawing this shit
# Размер клетки на доске
SQUARE_SIZE = 80

# Константы для игрового поля и клеток
WIDTH, HEIGHT = 800, 800
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

# Цвета
BROWN = (204, 119, 34)
YELLOW = (255, 204, 102)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)


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


# Создание окна Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Checkers Game")

# Функция для отрисовки доски и шашек

# Определение текущего выделенного квадрата
selected_piece = None


def draw_board(board):
    for row in range(8):
        for col in range(8):
            color = BROWN if (row + col) % 2 == 0 else WHITE
            pygame.draw.rect(
                screen,
                color,
                (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
            )

            piece = board[row][col]
            if piece != "_":
                piece_color = RED if piece == "W" else BLACK
                pygame.draw.circle(
                    screen,
                    piece_color,
                    (
                        col * SQUARE_SIZE + SQUARE_SIZE // 2,
                        row * SQUARE_SIZE + SQUARE_SIZE // 2,
                    ),
                    SQUARE_SIZE // 2 - 5,
                )

    pygame.display.update()


def get_row_col_from_mouse(pos):
    x, y = pos
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return row, col


def get_extra_move_from_player(row, col):
    # Ваш код для обработки ввода игрока (клика мыши) для выбора клетки для дополнительного удара
    # Возвращает кортеж с координатами начальной и конечной клеток удара, например, ((2, 5), (0, 3))
    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    new_row, new_col = get_row_col_from_mouse(pygame.mouse.get_pos())
                    return ((row, col), (new_row, new_col))


def main():
    global board, selected_piece

    ai = CheckersAI("W", max_depth=8)

    run = True

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Обрабатываем клик мыши
                if event.button == 1:
                    row, col = get_row_col_from_mouse(pygame.mouse.get_pos())
                    piece = board[row][col]

                    if piece != "_":
                        selected_piece = (row, col)
                        print("Selected piece:", selected_piece)

                elif event.button == 3 and selected_piece is not None:
                    # Если есть выделенная шашка и правый клик мыши, пробуем сделать ход
                    new_row, new_col = get_row_col_from_mouse(pygame.mouse.get_pos())

                    # Проверяем, что пытаемся переместиться на пустую клетку
                    if board[new_row][new_col] == "_":
                        move = (selected_piece, (new_row, new_col))
                        possible_moves = ai.get_possible_moves(board, "B")

                        # Проверяем, что ход является допустимым
                        if move in possible_moves:
                            old_board = board
                            board = ai.make_move(board, move)
                            print("Moved to:", (new_row, new_col))
                            ai.print_board(board)
                            if ai.has_extra_jumps(old_board, row, col, "B"):
                                # После вашего хода, проверяем наличие дополнительных ударов
                                while ai.has_extra_jumps(board, new_row, new_col, "B"):
                                    print("has extra jump")
                                    # Запрашиваем у игрока следующий удар
                                    extra_move = get_extra_move_from_player(
                                        new_row, new_col
                                    )
                                    extra_moves = ai.get_valid_moves_for_piece(
                                        board, new_row, new_col, "B"
                                    )
                                    mandatory_jumps = [
                                        move
                                        for move in extra_moves
                                        if abs(move[0][0] - move[1][0]) == 2
                                    ]
                                    if extra_move in mandatory_jumps:
                                        # Выполняем дополнительный удар
                                        board = ai.make_move(board, extra_move)
                                        new_row = extra_move[1][0]
                                        new_col = extra_move[1][1]
                                        print("Extra move to:", extra_move[1])
                                        ai.print_board(board)

                            # После вашего хода, делаем ход бота
                            bot_moves = ai.get_best_move(board)
                            if bot_moves:
                                bot_move = random.choice(bot_moves)
                                old_board = board
                                board = ai.make_move(board, bot_move)
                                print("Bot moved to:", bot_move[1])
                                ai.print_board(board)

                                if ai.has_extra_jumps(
                                    old_board, bot_move[0][0], bot_move[0][1], "W"
                                ):
                                    # После вашего хода, проверяем наличие дополнительных ударов
                                    while ai.has_extra_jumps(
                                        board, bot_move[1][0], bot_move[1][1], "W"
                                    ):
                                        print("bot has extra jump")
                                        # Запрашиваем у игрока следующий удар
                                        extra_move = get_extra_move_from_player(
                                            bot_move[1][0], bot_move[1][1]
                                        )
                                        extra_moves = ai.get_valid_moves_for_piece(
                                            board, bot_move[1][0], bot_move[1][1], "W"
                                        )
                                        mandatory_jumps = [
                                            move
                                            for move in extra_moves
                                            if abs(move[0][0] - move[1][0]) == 2
                                        ]
                                        if extra_move in mandatory_jumps:
                                            # Выполняем дополнительный удар
                                            board = ai.make_move(board, extra_move)
                                            new_row = extra_move[1][0]
                                            new_col = extra_move[1][1]
                                            print("bot extra move to:", extra_move[1])
                                            ai.print_board(board)

                                else:
                                    pass

        draw_board(board)

    pygame.quit()


if __name__ == "__main__":
    main()
