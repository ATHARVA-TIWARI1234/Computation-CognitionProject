import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import random


class RLSelector:
    def __init__(self, features, actions=["Yes", "No", "Maybe"], alpha=0.1, gamma=0.9):
        self.features = features
        self.actions = actions
        self.q_matrix = pd.DataFrame(0, index=features, columns=actions)  # Initialize Q-matrix
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor


    def choose_feature(self, explored_features, epsilon=0.1):
        left_features = [f for f in self.features if f not in explored_features]
        if not left_features:
            return None
        
        # Epsilon-Greedy exploration vs exploitation
        if random.uniform(0, 1) < epsilon:
            return random.choice(left_features) # random feature 
        else:
            q_sum = self.q_matrix.loc[left_features].sum(axis=1) 
            max_val = q_sum.max()
            optimal = q_sum[q_sum == max_val].index.tolist()
            return random.choice(optimal) # feature with the highest summed Q-value


    def adjust_q_value(self, feat, action, reward, next_feat=None):

        if action not in self.q_matrix.columns:
            self.q_matrix[action] = 0  # Initializing action column if it doesn't exist

        curr_q = self.q_matrix.loc[feat, action] # current Q-value 
        max_next_q = self.q_matrix.loc[next_feat].max() if next_feat else 0 # max Q-value for next feature 
        new_q = curr_q + self.alpha * (reward + self.gamma * max_next_q - curr_q) # Updating Q-value       
        self.q_matrix.loc[feat, action] = new_q # Storing updated Q-value 



class FoodAkinatorGame:
    def __init__(self, root_window):
        self.root_window = root_window
        self.root_window.title("AI Culinary Guesser")
        self.root_window.geometry("600x700")

        # Load character image
        self.character_image = Image.open("akinator.png")
        self.character_image = self.character_image.resize((150, 150), Image.Resampling.LANCZOS)
        self.character_image_tk = ImageTk.PhotoImage(self.character_image)

        # Food dataset
        self.culinary_items = {
            "pizza": {"Vegetarian": True, "Spicy": False, "Vegan": False, "Solid": True, "MainDish": True, "ServedHot": True},
            "sushi": {"Vegetarian": False, "Spicy": False, "Vegan": False, "Solid": True, "MainDish": True, "ServedHot": False},
            "ice cream": {"Vegetarian": True, "Spicy": False, "Vegan": False, "Solid": False, "MainDish": False, "ServedHot": False},
            "biryani": {"Vegetarian": False, "Spicy": True, "Vegan": False, "Solid": True, "MainDish": True, "ServedHot": True},
            "salad": {"Vegetarian": True, "Spicy": False, "Vegan": True, "Solid": False, "MainDish": False, "ServedHot": False},
            "burger": {"Vegetarian": True, "Spicy": True, "Vegan": False, "Solid": True, "MainDish": True, "ServedHot": True},
            "pasta": {"Vegetarian": True, "Spicy": False, "Vegan": False, "Solid": True, "MainDish": True, "ServedHot": False},
            "sandwich": {"Vegetarian": True, "Spicy": False, "Vegan": True, "Solid": True, "MainDish": False, "ServedHot": False},
            "tacos": {"Vegetarian": False, "Spicy": True, "Vegan": False, "Solid": True, "MainDish": True, "ServedHot": True},
            "noodles": {"Vegetarian": False, "Spicy": True, "Vegan": False, "Solid": True, "MainDish": True, "ServedHot": False},
            "quesadilla": {"Vegetarian": False, "Spicy": True, "Vegan": False, "Solid": True, "MainDish": True, "ServedHot": True},
            "smoothie": {"Vegetarian": True, "Spicy": False, "Vegan": True, "Solid": False, "MainDish": False, "ServedHot": False},
            "ramen": {"Vegetarian": False, "Spicy": True, "Vegan": False, "Solid": True, "MainDish": True, "ServedHot": True},
            "falafel": {"Vegetarian": True, "Spicy": False, "Vegan": True, "Solid": True, "MainDish": False, "ServedHot": False},
        }    
        self.feats = ["Vegetarian", "Spicy", "Vegan", "Solid", "MainDish", "ServedHot"]
        self.food_labels = list(self.culinary_items.keys())
        self.dataset = pd.DataFrame(self.culinary_items).T.astype(int)

        # Game attributes
        self.language_option = "English"
        self.difficulty_setting = "Hard"
        self.asked_feats = set()
        self.user_choices = {}
        self.feature_selector = RLSelector(self.feats)

        # Initialize UI
        self.init_ui()
        self.set_lang()
        self.set_diff()
        self.update_btn_txt()
        self.ask_question()

    def init_ui(self):
        # Display character image
        self.char_lbl = tk.Label(self.root_window, image=self.character_image_tk)
        self.char_lbl.pack(pady=10)

        # Question text
        self.query_lbl = tk.Label(self.root_window, text="", font=("Arial", 14), wraplength=400)
        self.query_lbl.pack(pady=20)

        # Answer buttons
        self.btn_frame = tk.Frame(self.root_window)
        self.btn_frame.pack(pady=10)
        self.btn_yes = tk.Button(self.btn_frame, text="Yes", command=lambda: self.handle_resp("yes"), width=10)
        self.btn_no = tk.Button(self.btn_frame, text="No", command=lambda: self.handle_resp("no"), width=10)
        self.btn_maybe = tk.Button(self.btn_frame, text="Maybe", command=lambda: self.handle_resp("maybe"), width=10)
        self.btn_yes.grid(row=0, column=0, padx=5)
        self.btn_no.grid(row=0, column=1, padx=5)
        self.btn_maybe.grid(row=0, column=2, padx=5)

    def get_responses(self):
        return {
            "English": {"yes": "Yes", "no": "No", "maybe": "Maybe", "predict": "I think the food is"},
        }

    def set_lang(self):
        langs = {"english": "English"}
        pref = simpledialog.askstring("Language Selection", "Choose a language (English):")
        self.lang = langs.get(pref.lower(), "English")

    def set_diff(self):
        levels = {"easy": "Easy", "hard": "Hard"}
        pref = simpledialog.askstring("Difficulty Selection", "Choose difficulty (Easy, Hard):")
        self.diff = levels.get(pref.lower(), "Easy")

    def update_btn_txt(self):
        texts = self.get_responses()[self.lang]
        self.btn_yes.config(text=texts["yes"])
        self.btn_no.config(text=texts["no"])
        self.btn_maybe.config(text=texts["maybe"])

    def next_feat(self):
        # first question 
        if not self.asked_feats:
            return "Vegetarian"

        if self.difficulty_setting == "Hard":
            return self.feature_selector.choose_feature(self.asked_feats)
        else:
            remaining_feats = [f for f in self.feats if f not in self.asked_feats]
            return random.choice(remaining_feats) if remaining_feats else None


    def ask_question(self):
        next_feat = self.next_feat()
        if next_feat:
            self.curr_feat = next_feat
            questions = {
                "Vegetarian": {"English": "Is it vegetarian?"},
                "Spicy": {"English": "Is it spicy?"},
                "Vegan": {"English": "Is it vegan?"},
                "Solid": {"English": "Is it solid?"},
                "MainDish": {"English": "Is it a main dish?"},
                "ServedHot": {"English": "Is it served hot?"}
            }
            question = questions.get(self.curr_feat, {}).get(self.lang, "")
            self.query_lbl.config(text=question)
        else:
            self.predict()

    def handle_resp(self, resp):
        # user's response 
        self.user_choices[self.curr_feat] = resp
        self.asked_feats.add(self.curr_feat)

        if resp == "yes":
            value = 1
        elif resp == "no":
            value = 0
        else:  # "maybe"
            value = 0.5

        # Update Q-values
        self.feature_selector.adjust_q_value(self.curr_feat, resp, value)

        # next question
        self.ask_question()


    def predict(self):
        food = self.get_food()
        messagebox.showinfo("Prediction", f"I think the food is {food}.")

    def get_food(self):
        feat_vector = [1 if self.user_choices.get(f, "no") == "yes" else 0 for f in self.feats]
        scores = {}
        for food, characteristics in self.culinary_items.items():
            scores[food] = sum(feat_vector[i] * characteristics[f] for i, f in enumerate(self.feats))
        return max(scores, key=scores.get) if scores else "unknown"


        

if __name__ == "__main__":
    root = tk.Tk()
    culinary_game = FoodAkinatorGame(root)
    root.mainloop()
