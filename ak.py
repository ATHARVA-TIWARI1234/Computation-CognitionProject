import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np

class FoodAkinatorGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Food Akinator")
        self.root.geometry("600x700")

        # Load Akinator image
        img = Image.open("akinator.png").resize((150, 150), Image.Resampling.LANCZOS)
        self.img = ImageTk.PhotoImage(img)

        # Food dataset
        self.foods = {
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
        self.reset_probs()

        # Display image
        tk.Label(root, image=self.img).pack(pady=10)

        # Question area
        self.q_label = tk.Label(root, text="", font=("Arial", 14), wraplength=400)
        self.q_label.pack(pady=20)

        # Answer buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        self.yes_btn = tk.Button(btn_frame, text="Yes", command=lambda: self.answer("yes"), width=10)
        self.yes_btn.grid(row=0, column=0, padx=5)
        self.no_btn = tk.Button(btn_frame, text="No", command=lambda: self.answer("no"), width=10)
        self.no_btn.grid(row=0, column=1, padx=5)
        self.maybe_btn = tk.Button(btn_frame, text="Maybe", command=lambda: self.answer("maybe"), width=10)
        self.maybe_btn.grid(row=0, column=2, padx=5)

        self.prog_label = tk.Label(root, text="", font=("Arial", 12))
        self.prog_label.pack(pady=5)

        self.curr_feat = None
        self.next_q()

    # Reset probabilities
    def reset_probs(self):
        self.probs = {f: 1 / len(self.foods) for f in self.foods}
        self.asked_feats = set()

    # Compute entropy
    def entropy(self, probs):
        return -sum(p * np.log2(p) for p in probs if p > 0)

    # information gain 
    def info_gain(self, feat):
        yes_p = sum(self.probs[f] for f, attrs in self.foods.items() if attrs[feat])
        no_p = 1 - yes_p

        if yes_p == 0 or no_p == 0:
            return 0

        # Entropy after yes/no split
        yes_ent = self.entropy([self.probs[f] / yes_p for f, attrs in self.foods.items() if attrs[feat]])
        no_ent = self.entropy([self.probs[f] / no_p for f, attrs in self.foods.items() if not attrs[feat]])

        return self.entropy(self.probs.values()) - (yes_p * yes_ent + no_p * no_ent)

    # next feature to ask 
    def pick_feat(self):
        feats_left = [f for f in self.feats if f not in self.asked_feats]
        if not feats_left:
            return None
        gains = {f: self.info_gain(f) for f in feats_left}
        return max(gains, key=gains.get)

    # Display next question
    def next_q(self):
        if max(self.probs.values()) < 0.9:
            self.curr_feat = self.pick_feat()
            if self.curr_feat:
                self.q_label.config(text=self.gen_q(self.curr_feat))
                self.asked_feats.add(self.curr_feat)
            else:
                self.guess()
        else:
            self.guess()

    # Generate question text
    def gen_q(self, feat):
        q_map = {
            "Vegetarian": "Is it vegetarian?",
            "Spicy": "Is it spicy?",
            "Vegan": "Is it vegan?",
            "Solid": "Is it solid in form?",
            "MainDish": "Is it a main course dish?",
            "ServedHot": "Is it served hot?",
        }
        return q_map.get(feat, f"Does it have {feat}?")

    # Update probabilities based on response
    def update_probs(self, resp):
        for f, attrs in self.foods.items():
            v = attrs.get(self.curr_feat)
            if resp == "yes":
                match = v
                self.probs[f] *= 1.0 if match else 0.0
            elif resp == "no":
                match = not v
                self.probs[f] *= 1.0 if match else 0.0
            elif resp == "maybe":
                self.probs[f] *= 0.5  # Decrease prob slightly for "maybe"

        tot = sum(self.probs.values())
        if tot == 0:
            self.reset_probs()
        else:
            for f in self.probs:
                self.probs[f] /= tot

    # Make a guess about the food
    def guess(self):
        best = max(self.probs, key=self.probs.get)
        prob = self.probs[best]

        if prob > 0.9:
            messagebox.showinfo("Guess", f"I guess your food is {best}!")
            self.root.quit()
        else:
            opts = [f for f, p in self.probs.items() if p > 0.1]
            opts_str = ", ".join(opts)
            self.q_label.config(text=f"Not sure. Could be: {opts_str}.")
            likely = max(opts, key=lambda f: self.probs[f])
            self.prog_label.config(text=f"Most likely: {likely}")

    # Handle user response
    def answer(self, resp):
        self.update_probs(resp)
        self.next_q()


# Main application
root = tk.Tk()
game = FoodAkinatorGame(root)
root.mainloop()
