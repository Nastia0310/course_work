import pygame
import random
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog
import torchvision
from torchvision import  datasets, models,  transforms
from PIL import Image
import requests
from io import BytesIO


pygame.init()

WHITE = (255, 255, 255)
BLACK = (40, 40, 40)
ORANGE = (235, 143, 56)
LIGHT_SAND = (255, 249, 233)
DARK_SAND = (251, 227, 167)
BLUE = (85, 93, 255)

main_screen_image = pygame.image.load('images/background_main.png')
score_screen_image = pygame.image.load('images/score_screen.png')
background_image = pygame.image.load('images/background.png')
easy_image = pygame.image.load('images/easy.png')
medium_image = pygame.image.load('images/medium.png')
hard_image = pygame.image.load('images/hard.png')

screen = pygame.display.set_mode((1440, 1024))
pygame.display.set_caption("quiz game")

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 256 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = MyNet()

model.load_state_dict(torch.load("model.pt",  map_location=torch.device('cpu')))
model.eval()



class Question:
    def __init__(self, question, options, correct_answer):
        self.question = question
        self.options = options
        self.correct_answer = correct_answer


base_questions = [
    Question("Какое животное издает звук \"мяу\"?", ["Собака", "Кошка", "Корова", "Лошадь"], 2),
    Question("Кто собирает орехи и прячет их в земле?", ["Белка", "Гепард", "Голубь", "Жираф"], 0),
    Question("Какое животное обычно создает паутины?", ["Паук", "Слон", "Верблюд", "Пингвин"], 0),
    Question("Какое животное вернsq друг человека?", ["Собака", "Лиса", "Волк", "Енот"], 0),
    Question("Какое насекомое  летает вокруг цветов?", ["Бабочка", "Крокодил", "Слон", "Носорог"], 0),
    Question("Какое домашнее животное несет яйца?", ["Хомяк", "Курица", "Гиена", "Медведь"], 1),
    Question("Какое животное дает нам молоко?", ["Корова", "Черепаха", "Олень", "Зебра"], 0),
    Question("Кто обладает длинным хоботом?", ["Слон", "Скорпион", "Комар", "Еж"], 0),
    Question("Кто используется для верховой езды?", ["Медведь", "Лошадь", "Гепард", "Лев"], 1),
    Question("Какое домашнее животное дает нам шерсть?", ["Волк", "Овца", "Опоссум", "Тигр"], 1),
    Question("Какое мелкое грызунчик живет в лесу?", ["Кенгуру", "Крокодил", "Белка", "Тукан"], 2),
    Question("Кто является символом верности?", ["Собака", "Слон", "Зебра", "Олень"], 0),
    Question("Какое насекомое появляется из кокона?", ["Бабочка", "Енот", "Бобр", "Белка"], 0),
    Question("Какое животное может быть черно-белым?", ["Дельфин", "Корова", "Акула", "Скат"], 1),
    Question("Какое животное славится большими ушами?", ["Волк", "Лев", "Слон", "Обезьяна"], 2),
    Question("Какое насекомое ловит насекомых в садах?", ["Варан", "Змея", "Паук", "Ящерица"], 2),
    Question("Кто дает шерсть и молоко?", ["Лев", "Кит", "Гепард", "Овца"], 3),

]

q = []


class QuizGame:
    def __init__(self, questions, num_questions, mode):
        self.questions = questions
        self.num_questions = num_questions
        self.current_questions = []
        self.current_question_index = 0
        self.score = 0
        self.mode = mode

    def start(self):
        self.current_questions = random.sample(self.questions, self.num_questions)
        self.current_question = self.current_questions[self.current_question_index]
        if self.mode == 'easy':
            self.show_easy_question()
        if self.mode == 'hard':
            self.show_hard_question()

    def show_easy_question(self):
        question_window = BaseQuestionWindow(self.current_question)
        answer = question_window.run()
        self.check_answer(answer)

    def show_hard_question(self):
        question_window = NeuralImageQuestion(model=model, question=self.current_question)
        answer = question_window.run()
        self.check_answer(answer)

    def check_answer(self, answer):
        if answer == self.current_question.correct_answer:
            self.score += 1

        if self.current_question_index < len(self.current_questions) - 1:
            self.current_question_index += 1
            self.current_question = self.current_questions[self.current_question_index]
            self.show_easy_question()
        else:
            self.game_over()

    def game_over(self):
        game_over_screen = GameOverScreen(self.score)
        game_over_screen.show()


class MainMenuScreen:
    def __init__(self):
        self.font = pygame.font.Font('fonts/OpenSans-SemiBold.ttf', 65)
        self.play_text = self.font.render("ИГРАТЬ", True, WHITE)

    def show(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if 280 <= mouse_x <= 697 and 540 <= mouse_y <= 668:
                        game_mode_screen = GameModeScreen()
                        game_mode_screen.show()
                        running = False

            screen.blit(main_screen_image, (0, 0))
            pygame.draw.rect(screen, ORANGE, (280, 540, 417, 128), 0, 12)
            screen.blit(self.play_text, (356, 560))
            pygame.display.flip()


class GameModeScreen:
    def __init__(self):
        self.title_font = pygame.font.Font('fonts/OpenSans-SemiBold.ttf', 70)
        self.font = pygame.font.Font('fonts/OpenSans-SemiBold.ttf', 45)
        self.title_text = self.font.render("Выберите режим игры", True, WHITE)
        self.easy_text = self.font.render("классический", True, WHITE)
        self.hard_text = self.font.render("расширенный", True, WHITE)
        self.selected_mode = None

    def show(self):
        global game
        screen = pygame.display.set_mode((1440, 1024))
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if 361 < mouse_pos[1] < 904:
                        if 174 < mouse_pos[0] < 646:
                            self.selected_mode = "easy"
                            running = False

                        elif 768 < mouse_pos[0] < 1240:
                            self.selected_mode = "hard"
                            running = False

            screen.blit(background_image, (0, 0))
            self.draw_menu(screen)
            pygame.display.flip()
            clock.tick(60)

        if self.selected_mode:
            if self.selected_mode == "easy":
                game = QuizGame(base_questions, 10, 'easy')

            elif self.selected_mode == "hard":
                game = QuizGame(base_questions, 10, 'hard')

            game.start()

    def draw_menu(self, screen):
        title_font = pygame.font.Font('fonts/OpenSans-SemiBold.ttf', 70)
        text = title_font.render("Выберите уровень сложности", True, WHITE)
        screen.blit(text, (174, 187))

        font = pygame.font.Font('fonts/OpenSans-SemiBold.ttf', 45)

        pygame.draw.rect(screen, ORANGE, (174, 361, 472, 543), 0, 12)
        text = font.render("классический", True, WHITE)
        screen.blit(text, (251, 785))
        screen.blit(easy_image, (265, 437))

        pygame.draw.rect(screen, ORANGE, (768, 361, 472, 543), 0, 12)
        text = font.render("расширенный", True, WHITE)
        screen.blit(text, (852, 785))
        screen.blit(hard_image, (871, 437))

class BaseQuestionWindow:
    def __init__(self, question):
        self.question = question

    def run(self):
        screen = pygame.display.set_mode((1440, 1024))
        clock = pygame.time.Clock()

        running = True
        answer = None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if 85 <= mouse_pos[0] <= 692:
                        if 517 <= mouse_pos[1] <= 646:
                            answer = 0
                            running = False
                        elif 688 <= mouse_pos[1] <= 817:
                            answer = 2
                            running = False
                    elif 737 <= mouse_pos[0] <= 1344:
                        if 517 <= mouse_pos[1] <= 646:
                            answer = 1
                            running = False
                        elif 688 <= mouse_pos[1] <= 817:
                            answer = 3
                            running = False

            screen.blit(background_image, (0, 0))
            self.draw_question(screen)
            pygame.display.flip()
            clock.tick(60)

        return answer

    def draw_question(self, screen):
        font = pygame.font.Font('fonts/OpenSans-SemiBold.ttf', 45)
        text = font.render(self.question.question, True, BLACK)
        pygame.draw.rect(screen, LIGHT_SAND, (85, 150, 1259, 271), 0, 12)
        screen.blit(text, (137, 193))

        options = self.question.options
        for i, option in enumerate(options):
            text = font.render(option, True, BLACK)
            if i == 0:
                pygame.draw.rect(screen, LIGHT_SAND, (85, 517, 607, 129), 0, 12)
                pygame.draw.circle(screen, DARK_SAND, (136, 582), 22)
                screen.blit(text, (182, 551))
            elif i == 1:
                pygame.draw.rect(screen, LIGHT_SAND, (737, 517, 607, 129), 0, 12)
                pygame.draw.circle(screen, DARK_SAND, (789, 582), 22)
                screen.blit(text, (831, 551))
            elif i == 2:
                pygame.draw.rect(screen, LIGHT_SAND, (85, 688, 607, 129), 0, 12)
                pygame.draw.circle(screen, DARK_SAND, (137, 753), 22)
                screen.blit(text, (182, 722))
            else:
                pygame.draw.rect(screen, LIGHT_SAND, (737, 688, 607, 129), 0, 12)
                pygame.draw.circle(screen, DARK_SAND, (789, 753), 22)
                screen.blit(text, (831, 722))


class GameOverScreen:
    def __init__(self, score):
        self.font = pygame.font.Font('fonts/OpenSans-SemiBold.ttf', 75)
        self.title_text = self.font.render("Игра окончена!", True, WHITE)
        self.text = self.font.render("Ваш счет: {}/10".format(score), True, WHITE)
        self.button_font = pygame.font.Font('fonts/OpenSans-SemiBold.ttf', 60)
        self.play_again_text = self.button_font.render("ЕЩЁ РАЗ", True, WHITE)

    def show(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if 856 <= mouse_x <= 1220 and 586 <= mouse_y <= 714:
                        game_mode_screen = GameModeScreen()
                        game_mode_screen.show()
                        running = False

            screen.blit(score_screen_image, (0, 0))
            screen.blit(self.title_text, (743, 309))
            screen.blit(self.text, (765, 415))
            pygame.draw.rect(screen, ORANGE, (856, 586, 364, 128), 0, 12)
            screen.blit(self.play_again_text, (909, 609))
            pygame.display.flip()


class NeuralImageQuestion:
    def __init__(self, model, question):
        self.question = question
        self.model = model
        self.correct_answer = self.question.correct_answer

    def run(self):
        screen = pygame.display.set_mode((1440, 1024))
        clock = pygame.time.Clock()

        running = True
        answer = None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if 90 <= mouse_pos[0] <= 1350 and 736 <= mouse_pos[1] <= 865:
                        answer = self.choose_image_and_classify()
                        print(type(answer))
                        running = False

            screen.blit(background_image, (0, 0))
            self.draw_question(screen)
            pygame.display.flip()
            clock.tick(60)

        if answer is not None and answer == self.correct_answer:
            return 1
        else:
            return 0

    def choose_image_and_classify(self):
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg")],
        )

        if file_path:
            image = Image.open(file_path).convert("RGB")
            transformed_image = self.transform_image(image)
            prediction = self.model(transformed_image)
            return prediction
        else:
            return None

    def transform_image(self, image):
        transform = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.ToTensor(),
        ])
        return transform(image)

    def draw_question(self, screen):
        font = pygame.font.Font('fonts/OpenSans-SemiBold.ttf', 45)
        text = font.render(self.question.question, True, BLACK)
        pygame.draw.rect(screen, LIGHT_SAND, (85, 150, 1259, 271), 0, 12)
        screen.blit(text, (137, 193))
        pygame.draw.rect(screen, LIGHT_SAND, (90, 736, 1259, 129), 0, 12)



main_menu_screen = MainMenuScreen()
main_menu_screen.show()

pygame.quit()
