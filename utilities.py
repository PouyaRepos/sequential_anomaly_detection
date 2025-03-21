import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from itertools import product
import math
from collections import defaultdict
import random

class NoisyLearningAlgorithm:
    def __init__(self, n_budget, alpha, lambda_, feature_dimension,
                 initial_depth_level = 2, delta= 0.05, beta= 1,
                 c1= 1, c3= 1, error_rate= 0.0,
                 B_l_alpha= 0.5, max_depth_level = 10, random_state= None ):

        """Initialize the noisy learning algorithm."""
        self.n = n_budget  # labeling budget
        self.alpha = alpha  # Hölder smoothness (≤ 1)
        self.lambda_ = lambda_  # Hölder constant
        self.l = initial_depth_level # Initial depth level of dyadic grid
        self.k = feature_dimension  # feature dimensions
        self.delta = delta  # confidence leveal
        self.beta = beta 
        self.c1 = c1
        self.c3 = c3
        self.error_rate = error_rate # optional error rate during labeling
        self.B_l_alpha = B_l_alpha
        self.random_state = random_state
        self.max_depth_level = max_depth_level

    
    # Function to assign points to dyadic grid cell
    def _get_dyadic_grid(self, points):
        # Organize points into dyadic cells up to level L
        dyadic_grid = {l: {} for l in range(1, self.max_depth_level + 1)}
        # print(f'empty: {dyadic_grid}')
        for point in points:
            for l in range(1, self.max_depth_level + 1):
                
                cell = tuple(int(point[i] * (2**l)) for i in range(len(point)))
                # print(f' level and cell: {l}, {cell}')
                if cell not in dyadic_grid[l]:
                    dyadic_grid[l][cell] = []
                dyadic_grid[l][cell].append(tuple(point))
        
        #Ensure all levels contain an entry for each expected cell
        for l in dyadic_grid:
            dyadic_grid[l] = {cell: dyadic_grid[l].get(cell, []) for cell in dyadic_grid[l]}  

        return dyadic_grid
    

    def get_children_cells(self, parent_cell_index):
        """
        Given a parent cell index at level l, return a list of 2^k child cell indices at level l+1.
        
        :param parent_cell_index: Tuple representing the parent cell index (m1, m2, ..., mk).
        :return: List of tuples representing child cell indices at level l+1.
        """
        children = []
        for offsets in product([0, 1], repeat=self.k):
            child_index = tuple(int(2 * parent_cell_index[i] + offsets[i]) for i in range(self.k))  # Convert np.int64 to int
            children.append(child_index)  #Ensure a list of tuples

        return children  # matches the format of `list(dyadic_grid[1].keys())`



    def _sample_from_cell(self, level, cell, t_l_alpha, true_labels, data, dyadic_grid):
        """
        Sample t_l_alpha points from a given dyadic cell at level l_current.
        
        :param l_current: Current depth level of the dyadic grid.
        :param cell: Tuple representing the cell's index in the dyadic grid.
        :param t_l_alpha: Number of points to sample from the cell.
        :param true_labels: True labels of the dataset.
        :param data: The dataset containing feature vectors.
        
        :return: List of sampled labels from the specified cell.
        """
        cell_key = tuple(cell)  # Convert numpy array to tuple for dictionary lookup

        # Check if level exists
        if level not in dyadic_grid:
            print(f"Warning: Level {level} not found in dyadic_grid. Skipping sampling.")
            return []

        # Check if the cell exists in the given level
        if cell_key not in dyadic_grid[level]:
            print(f"Warning: Cell {cell_key} not found in dyadic_grid[{level}]. Skipping sampling.")
            return []

        cell_points = dyadic_grid[level][cell_key]

        # If there are no points in the cell, return an empty list
        if not cell_points:
            print(f"Warning: Cell {cell_key} exists but has no points in dyadic_grid[{level}]. Skipping sampling.")
            return []

        # Sample points (if more available, randomly select t_l_alpha; else, return all)
        sampled_points = random.sample(cell_points, min(len(cell_points), int(t_l_alpha)))

        # Get corresponding labels
        sampled_labels = [true_labels[np.where((data == point).all(axis=1))[0][0]] for point in sampled_points]

        return sampled_labels


    def _b_l_alpha(self, level):
        return self.lambda_ * (self.k ** (self.alpha / 2)) * (2 ** (-level * self.alpha))
    
    def _delta_l_alpha(self, level):
        return self.delta * (2 ** (-level * (self.k + 1)))

    def _t_l_alpha(self, level):
        return int(math.ceil(np.log2(1 / self._delta_l_alpha(level)) / (2 * self._b_l_alpha(level) ** 2)))

    def _B_l_alpha(self, level):
        """Compute B_{l,α} for the confidence bound."""
        deviation = np.sqrt(np.log2(1 / self._delta_l_alpha(level)) / (2 * self._t_l_alpha(level)))
        return 2 * (deviation + self._b_l_alpha(level))

    def run(self, data, true_labels):
        """Run the modified Algorithm 2 on the data."""
        self.S0, self.S1, self.ac = set(), set(), set()  # labeled sets for classes 0 and 1
        l_current = self.l  # level. Use local variable instead of modifying self.l immediately

        dyadic_grid = self._get_dyadic_grid(data) # all cells in all levels and points within them {level:{(cell):[points]}}
        # print(f'dyadic_grid.keys(): {dyadic_grid.keys()}')
        all_cell_indices = {}
        for level_ in dyadic_grid:
            all_cell_indices.setdefault(level_, []).extend(dyadic_grid[level_].keys())

        # print(all_cell_indices)
        # print(f"last level's cell: {dyadic_grid[l_current].keys()}")
        self.active_cells=  list(dyadic_grid[l_current].keys()) # Initial active cells
        # print(f'dyadic_grid[{l_current}].keys(): {self.active_cells}')
        # print(f'self.active_cells: {self.active_cells}')
        t = 2 ** self.k * self._t_l_alpha(l_current)  # initial sample count
        results = [] # to keep track of S0, S1, and A_l over all levels


        while t <= self.n:
            # print(f't: {t} and level: {l_current}')
            result = {"level":[], "S0": [], "S1": [], "A_{l+1}": []}
            new_active_cells = set()
            # print(f' length of active_cells: {len(self.active_cells)}')


            for cell in self.active_cells:

                # Handel empty cells situation. Assign empty cells randomly to S0 or S1.
                cell_key = tuple(cell)  # Ensure tuple format
                if cell_key not in dyadic_grid[l_current]:  # Check if the cell is empty
                    assigned_label = random.choice(["S0", "S1"])
                    # print(f"Empty cell {cell_key} at level {l_current} assigned to {assigned_label}.")
                    (self.S1 if assigned_label == "S1" else self.S0).add((cell_key, l_current))
                    result[assigned_label].append(cell)
                    continue  # Skip to the next cell


                # Sample t_{l,α∧1} samples from the entire cell (modified)
                # print(f'_t_l_alpha: {self._t_l_alpha(l_current)}')
                sampled_labels = self._sample_from_cell(l_current, cell, self._t_l_alpha(l_current), true_labels, data, dyadic_grid)
                # print(f'len samples: {len(sampled_labels)}')
                # print(f'samples: {sampled_labels}')

                
                eta_hat = np.mean(sampled_labels)  # empirical estimate of η(x_C)
            
                if self.B_l_alpha is None: # Checks for user's given B_l_alpha
                    self.B_l_alpha = self._B_l_alpha(l_current)

                # print(f'condition for geting chidren: {abs(eta_hat - 0.5) <= self.B_l_alpha}')
                if abs(eta_hat - 0.5) <= self.B_l_alpha:
                    # Refine the grid: add subcells to next depth
                    # print(f'cell for get_children_cells: {cell}' )
                    subcells = self.get_children_cells(cell)
                    self.ac.add((tuple(subcells), l_current))
                    # print(f'subcells type: {subcells}')
                    # new_active_cells.update(subcells)
                    new_active_cells.update(subcells)
                        
                    # print(f' new_active_cells: {new_active_cells}')

                    # result["A_{l+1}"].extend(new_active_cells)
                    result["A_{l+1}"].extend(subcells)

                
                else:
                    # Label the cell
                    (self.S1 if eta_hat >= 0.5 else self.S0).add((tuple(cell), l_current))
                    # print(f'self.S0: {self.S0}, and self.S1:{self.S1}')
                    result["S1"].append(cell) if eta_hat >= 0.5 else result["S0"].append(cell)
            result['level'].append(l_current) 
            # print(f'result in for: {result}')       
            results.append(result)
            # print(f'results in for: {results}')

            if not new_active_cells:
                print("No more active cells. Breaking out of the loop.")
                break

            self.active_cells = new_active_cells
            
            l_current += 1
            # print(f'len(self.active_cells) * self._t_l_alpha(l_current): {len(self.active_cells)} and  {self._t_l_alpha(l_current)}')
            t += len(self.active_cells) * self._t_l_alpha(l_current)
            # print(f't updated: {t}')
            # print(f' down while cond: {t <= self.n}')
            # print("--"* 50)

            # Check if budget is exhausted (t >= n) assign remaining active cells randomly to S0 and S1
            if t >= self.n:
                # print(f't has been exhuasted. Remaining active cells will be assigned randomly')
                result = {"level":[], "S0": [], "S1": [], "A_{l+1}": []}
            # Assign remaining active cells at the L level to classes randomly
                for cell in self.active_cells:

                    # selected_set_name = "S1" if np.random.rand() < 0.5 else "S0"
                    # selected_set = self.S1 if selected_set_name == "S1" else self.S0

                    # # selected_set.add((tuple(cell), l_current))
                    # # result[selected_set_name].append(cell)
                    # selected_set.add(cell)
                    # result[selected_set_name].append(selected_set)
                    # print(f' l_current for t>= n: {l_current}')
                    (self.S1 if np.random.rand() < 0.5 else self.S0).add((tuple(cell), l_current))
                    result["S1"].append(cell) if np.random.rand() < 0.5 else result["S0"].append(cell)
                result['level'].append(l_current) 
                # print(f'result for t >=n: {result}')       
                results.append(result)
                # print(f'resultsss for t >=n: {results}')
                break
        
        dct = defaultdict(list)
        for value, key in list(self.S0):
            dct[key].append(value)

        # Convert to a regular dictionary if needed
        self.s0_final = dict(dct)

        dct = defaultdict(list)
        for value, key in list(self.S1):
            dct[key].append(value)

        # Convert to a regular dictionary if needed
        self.s1_final = dict(dct)

        dct = defaultdict(list)
        for value, key in list(self.ac):
            dct[key].append(value)

        # Convert to a regular dictionary if needed
        self.ac_final = dict(dct)

        # print(f'final self.S0: {self.s0_final}, and self.S1:{self.s1_final}')
        # print(f'final self.S1: {self.s1_final}')

        # print(f'final self.S0: {self.s0_final}')

        # print(f'final self.ac: {self.ac_final}')


        # result['level'].append(l_current)        
        # results.append(result)
        # print(f'results: {results}')
        # Construct the classifier
        return self.s0_final, self.s1_final, self.ac, results

    def predict(self, points_list):
        """
        Predict whether points belong to a cell in Ss1.
        
        Parameters:
        - points_list: A list of tuples or arrays representing points in [0, 0.999] space
        - Ss1: Dictionary where keys are depth levels and values are lists of cells at that level
        
        Returns:
        - A list of 1s and 0s indicating whether each point belongs to a cell in Ss1
        """
        results = []
        
        for points in points_list:
            # Convert point to tuple if it's not already
            points = tuple(points) if not isinstance(points, tuple) else points
            
            # Check each level in Ss1, starting from the deepest level
            belongs_to_cell = 0
            for level in sorted(self.s1_final.keys(), reverse=True):
                # Calculate the cell coordinates for this level
                cell_coords = tuple(int(coord * (2**level)) for coord in points)
                # Check if the cell exists in Ss1 at this level
                if cell_coords in self.s1_final[level]:
                    belongs_to_cell = 1
                    break
            results.append(belongs_to_cell)
        
        return results

    def plot_dyadic_grid(self, X, y):
        """
        Plots the dyadic grid based on the feature dimension:
        - If 2D, it plots a 2D representation.
        - If 3D, it plots an interactive 3D visualization.
        - Otherwise, it prints a message.

        Parameters:
        - s0: Dictionary with levels as keys and lists of cell coordinates as values (class 0)
        - s1: Dictionary with levels as keys and lists of cell coordinates as values (class 1)
        - data: Array of data points
        - labels: Array of true labels (0 or 1)
        """
        feature_dimension = self.k
        alpha_cells = 0.3
        alpha_points = 0.5
        data_0 = X[y == 0]
        data_1 = X[y == 1]

        if feature_dimension == 2:
            # 2D Plot
            fig, ax = plt.subplots(figsize=(8, 8))

            def draw_cell(ax, cell, level, color):
                cell_size = 1 / (2**level)
                x_start = cell[0] * cell_size
                y_start = cell[1] * cell_size
                rect = plt.Rectangle((x_start, y_start), cell_size, cell_size, linewidth=1, 
                                    edgecolor=color, facecolor=color, alpha=alpha_cells)
                ax.add_patch(rect)

            for level, cells in self.s0_final.items():
                for cell in cells:
                    draw_cell(ax, cell, level, 'red')

            for level, cells in self.s1_final.items():
                for cell in cells:
                    draw_cell(ax, cell, level, 'blue')


            ax.scatter(data_0[:, 0], data_0[:, 1], color='red', marker='o', alpha=alpha_points, label='Class 0')
            ax.scatter(data_1[:, 0], data_1[:, 1], color='blue', marker='o', alpha=alpha_points, label='Class 1')

            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_title("2D Dyadic Grid of Classification")
            ax.legend()
            plt.grid(False)
            plt.show()

        elif feature_dimension == 3:
            # 3D Plot
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')

            colors = {'s0': 'blue', 's1': 'red'}

            ax.scatter(data_0[:, 0], data_0[:, 1], data_0[:, 2], c='blue', alpha=alpha_points, s=20, label='Class 0')
            ax.scatter(data_1[:, 0], data_1[:, 1], data_1[:, 2], c='red', alpha=alpha_points, s=20, label='Class 1')

            all_levels = set(self.s0_final.keys()).union(self.s1_final.keys())

            for level in all_levels:
                cell_size = 1.0 / (2 ** level)

                if level in self.s0_final:
                    for cell in self.s0_final[level]:
                        x, y, z = [coord * cell_size for coord in cell]
                        ax.bar3d(x, y, z, cell_size, cell_size, cell_size, color=colors['s0'], alpha=alpha_cells, edgecolor='none')

                if level in self.s1_final:
                    for cell in self.s1_final[level]:
                        x, y, z = [coord * cell_size for coord in cell]
                        ax.bar3d(x, y, z, cell_size, cell_size, cell_size, color=colors['s1'], alpha=alpha_cells, edgecolor='none')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)

            legend_elements = [
                Patch(facecolor=colors['s0'], alpha=alpha_cells, label='Class 0 Cells'),
                Patch(facecolor=colors['s1'], alpha=alpha_cells, label='Class 1 Cells'),
                Line2D([0], [0], marker='o', color='w', label='Data (Label 0)', markerfacecolor='blue', markersize=5),
                Line2D([0], [0], marker='o', color='w', label='Data (Label 1)', markerfacecolor='red', markersize=5)
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            plt.title('3D Dyadic Grid of Data Points Classification')
            plt.show()

        else:
            print(f"Feature dimension {feature_dimension} is not supported for visualization.")

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates a model by computing accuracy, AUC-ROC, and an aggregated classification report.

        Parameters:
        - model: The trained model.
        - X_test: Test feature set.
        - y_test: True labels for the test set.
        """
        y_pred = np.array(self.predict(X_test))
        y_pred_prob = np.array(y_pred, dtype=float)  

        accuracy = np.mean(y_pred == y_test)
        auc_roc = roc_auc_score(y_test, y_pred_prob) if len(set(y_test)) > 1 else 0.5
        
        # Fix: Use zero_division=0 to prevent warnings
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        print("Model:", self.__class__.__name__)
        print(f"Mean Accuracy: {accuracy:.4f}")
        print(f"Mean AUC-ROC: {auc_roc:.4f}")
        print(f'report: {pd.DataFrame(report)}')
        # # Fix: Print full output
        # import json
        # print(json.dumps(report, indent=4))  # Pretty-print for full view
        """
        Plot per-class classification metrics (Precision, Recall, F1-Score, AUC-ROC).
        
        Parameters:
        - report: The classification report dictionary.
        - auc_roc: Computed AUC-ROC score (float).
        """
        labels = ["precision", "recall", "f1-score"]
        
        class_0 = [report["0"][m] for m in labels]
        class_1 = [report["1"][m] for m in labels]

        # Append AUC-ROC (same for both classes in binary classification)
        class_0.append(auc_roc)
        class_1.append(auc_roc)

        x = ["precision", "recall", "f1-score", "auc_roc"]
        width = 0.3

        fig, ax = plt.subplots()
        ax.bar(np.arange(len(x)) - width/2, class_0, width, label="Class 0")
        ax.bar(np.arange(len(x)) + width/2, class_1, width, label="Class 1")

        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(x)
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Metrics")
        ax.legend()

        plt.show()
        return


