#!/usr/bin/env python3
import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Set
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress

# Add src to path if running from root
sys.path.append(os.path.abspath("src"))

from explainers_lib.explainers.native.wachter import WachterExplainer
from explainers_lib.explainers.native.growing_spheres import GrowingSpheresExplainer
from explainers_lib.explainers.native.face import FaceExplainer
from explainers_lib.explainers.dice.dice import DiceExplainer
from explainers_lib.explainers.celery_explainer import AlibiCFProto, AlibiCFRL
# from explainers_lib.explainers.celery_explainer import ActionableRecourseExplainer # Currently broken
from explainers_lib.aggregators import Pareto, IdealPoint, BalancedPoint, TOPSIS, DensityBased, ScoreBasedAggregator
from explainers_lib.datasets import Dataset
from explainers_lib.ensemble import Ensemble, cfs_group_by_original_data
from explainers_lib.model import TorchModel
from explainers_lib.counterfactual import Counterfactual

console = Console()

RESULTS_FILE = "experiments/titanic_results.json"

def clear_screen():
    console.clear()

def get_formatted_changes(ds: Dataset, original_vector: np.ndarray, cf_vector: np.ndarray) -> str:
    # Calculate impact
    diff = np.abs(cf_vector - original_vector)
    changes = []
    
    # Continuous
    # In ColumnTransformer, 'num' comes first if it was defined first in the list.
    # Dataset.get_preprocessor defines [('num', ...), ('cat', ...)]
    # So indices 0 to len(cont) are continuous.
    
    for i, name in enumerate(ds.continuous_features):
        impact = diff[i]
        if impact > 1e-3:
            changes.append({'name': name, 'impact': impact, 'type': 'num'})
    
    # Categorical
    current_idx = len(ds.continuous_features)
    
    try:
        # Accessing the OneHotEncoder from the pipeline inside ColumnTransformer
        # Structure: ColumnTransformer -> 'cat' (Pipeline) -> 'onehot' (OneHotEncoder)
        cat_categories = ds.preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_
        
        for i, name in enumerate(ds.categorical_features):
            n_cats = len(cat_categories[i])
            slice_diff = diff[current_idx : current_idx + n_cats]
            impact = np.sum(slice_diff)
            if impact > 1e-3:
                changes.append({'name': name, 'impact': impact, 'type': 'cat'})
            current_idx += n_cats
    except Exception as e:
        # Fallback if accessing internals fails, though it shouldn't with the known library structure
        # Just list changes based on inverse transform if impact calculation fails
        pass
        
    # Sort by impact desc
    changes.sort(key=lambda x: x['impact'], reverse=True)
    
    # Get readable values
    orig_df = ds.inverse_transform(np.array([original_vector]))
    cf_df = ds.inverse_transform(np.array([cf_vector]))
    
    parts = []
    
    # If we couldn't calculate impact for categories or list is empty, fallback to simple diff
    if not changes:
        # Simple fallback check
        for col in orig_df.columns:
            old_val = orig_df[col].iloc[0]
            new_val = cf_df[col].iloc[0]
            if old_val != new_val:
                 changes.append({'name': col, 'impact': 0}) # Dummy impact

    for i, change in enumerate(changes):
        name = change['name']
        old_val = orig_df[name].iloc[0]
        new_val = cf_df[name].iloc[0]
        
        # Formatting values
        if isinstance(old_val, float) or isinstance(new_val, float):
            try:
                val_str = f"{float(old_val):.2f}->{float(new_val):.2f}"
            except:
                val_str = f"{old_val}->{new_val}"
        else:
            val_str = f"{old_val}->{new_val}"
            
        item_str = f"[bold]{name}[/bold]: {val_str}"
        
        parts.append(item_str)
        
    return ", ".join(parts)

def load_titanic_data():
    with console.status("[bold green]Loading Titanic Dataset..."):
        # TODO: Check if local file exists to avoid download
        url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
        try:
            df = pd.read_csv(url)
        except Exception as e:
            console.print(f"[bold red]Error downloading dataset:[/bold red] {e}")
            sys.exit(1)

        df = df.drop(['Name'], axis=1)
        categorical_features = ['Sex', 'Pclass']
        numerical_features = ['Age', 'Fare', 'Parents/Children Aboard', 'Siblings/Spouses Aboard']
        target = 'Survived'
        
        X = df.drop(target, axis=1)
        y = df[target]
        
        # Consistent split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        ds = Dataset(X_test, y_test.values, X_test.columns.tolist(), categorical_features=categorical_features, continuous_features=numerical_features)
        return ds

def load_model(ds):
    with console.status("[bold green]Loading Model..."):
        model_path = "experiments/models/titanic_classifier.pt"
        if not os.path.exists(model_path):
            console.print(f"[bold red]Model not found at {model_path}[/bold red]")
            sys.exit(1)
            
        with open(model_path, "rb") as f:
            model_data = f.read()
        
        model = TorchModel.deserialize(model_data)
        return model

def configure_explainers():
    explainers = []
    console.print(Panel("Configure Ensemble Explainers", style="bold blue"))
    
    options = {
        "1": ("WachterExplainer", WachterExplainer),
        "2": ("GrowingSpheresExplainer", GrowingSpheresExplainer),
        "3": ("FaceExplainer", FaceExplainer),
        "4": ("DiceExplainer", DiceExplainer),
        "5": ("AlibiCFProto", AlibiCFProto),
        "6": ("AlibiCFRL", AlibiCFRL)
    }
    
    console.print("Available Explainers:")
    for k, v in options.items():
        console.print(f"[{k}] {v[0]}")
        
    choices = Prompt.ask("Enter numbers of explainers to include (comma separated)", default="1,2,3")
    selected_keys = [k.strip() for k in choices.split(",") if k.strip() in options]
    
    for k in selected_keys:
        name, cls = options[k]
        if name == "GrowingSpheresExplainer":
            max_radius = IntPrompt.ask(f"[{name}] Max Radius", default=16)
            num_cfs = IntPrompt.ask(f"[{name}] Num CFs", default=5)
            for _ in range(num_cfs):
                explainers.append(cls(max_radius=max_radius))
        elif name == "DiceExplainer":
            num_cfs = IntPrompt.ask(f"[{name}] Num CFs", default=8)
            explainers.append(cls(num_cfs=num_cfs))
        elif name == "WachterExplainer":
             explainers.append(cls(lambda_param=[0.1, 0.5, 1, 5, 10, 50, 100]))
        elif name == "FaceExplainer":
             explainers.append(cls(fraction=1.0))
        else:
            explainers.append(cls())
            
    return explainers

def save_result(picked_cf: Dict):
    data = []
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            pass # Start fresh if corrupt
            
    data.append(picked_cf)
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    # console.print(f"[dim]Saved to {RESULTS_FILE}[/dim]") # Removed as requested

def view_results():
    if not os.path.exists(RESULTS_FILE):
        console.print("[bold red]No results file found.[/bold red]")
        return

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)
        
    if not data:
        console.print("[bold yellow]Results file is empty.[/bold yellow]")
        return

    # Load dataset for inverse transform
    ds = load_titanic_data()

    # Group by original data bytes
    grouped_data = {}
    for entry in data:
        # Use a tuple of the list for hashing
        key = tuple(entry['original_data'])
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(entry)

    console.print(Panel(f"Saved Counterfactuals (Total: {len(data)})", style="bold green"))
    Prompt.ask("Press Enter to view results") # Pause before first group
    
    # Convert grouped_data dictionary to a list of (original_data_key, entries) for indexed iteration
    grouped_instances_list = list(grouped_data.items())
    total_unique_instances = len(grouped_instances_list)
    current_instance_idx = 0

    while current_instance_idx < total_unique_instances:
        original_data_tuple, entries = grouped_instances_list[current_instance_idx]
        
        clear_screen() # Clear screen for each group

        # Display original instance data
        original_data_np = np.array(original_data_tuple)
        orig_df = ds.inverse_transform(np.array([original_data_np]))
        
        console.print(Panel(f"Viewing Instance {current_instance_idx + 1}/{total_unique_instances}:", style="bold blue"))
        console.print(f"Original Prediction: [bold cyan]{entries[0]['original_class']}[/bold cyan]") # Assuming all CFs in group have same original_class
        console.print("Original Data:")
        console.print(orig_df.to_string(index=False))


        table = Table(title=f"Counterfactuals for this instance")
        table.add_column("ID", justify="right", style="cyan", no_wrap=True)
        table.add_column("Changes", style="yellow")
        table.add_column("Orig -> Target", justify="center")
        table.add_column("Explainer", style="magenta")
        table.add_column("Selected By", style="green")

        for i, entry in enumerate(entries):
            orig = np.array(entry['original_data'])
            cf = np.array(entry['cf_data'])
            
            change_str = get_formatted_changes(ds, orig, cf)
            
            table.add_row(
                str(i+1),
                change_str,
                f"{entry['original_class']} -> {entry['target_class']}",
                entry['explainer'],
                ", ".join(entry['selected_by'])
            )
        
        console.print(table)
        console.print("\n[b]Options:[/b] [yellow]n[/yellow] (next), [yellow]p[/yellow] (previous), [red]q[/red] (quit view)")
        choice = Prompt.ask("Action")

        if choice.lower() == 'n':
            current_instance_idx = (current_instance_idx + 1) % total_unique_instances
        elif choice.lower() == 'p':
            current_instance_idx = (current_instance_idx - 1 + total_unique_instances) % total_unique_instances
        elif choice.lower() == 'q':
            break
        else:
            console.print("[red]Unknown command[/red]")
            time.sleep(0.5)

def run_experiment():
    ds = load_titanic_data()
    model = load_model(ds)
    explainers = configure_explainers()
    batch_size = IntPrompt.ask("Enter batch size", default=20)
    
    if not explainers:
        console.print("[red]No explainers selected. Aborting.[/red]")
        return

    ensemble = Ensemble(model, explainers)
    
    # Train/Fit explainers
    with console.status("[bold blue]Fitting Explainers... (this may take a moment)"):
        ensemble.fit(ds)
        
    selectors = {
        "Density": DensityBased(),
        "Pareto": Pareto(),
        "IdealPoint": IdealPoint(),
        "BalancedPoint": BalancedPoint(),
        "TOPSIS": TOPSIS()
    }
    
    # Fit score-based selectors
    for name, sel in selectors.items():
        if isinstance(sel, ScoreBasedAggregator):
            sel.fit(model, ds)

    total_instances = len(ds.data)
    current_idx = 0
    
    while current_idx < total_instances:
        end_idx = min(current_idx + batch_size, total_instances)
        console.print(Panel(f"Processing Batch {current_idx} - {end_idx} of {total_instances}", style="bold green"))
        
        batch_ds = ds[current_idx:end_idx]
        
        with console.status(f"[bold blue]Generating CFs for batch {current_idx}-{end_idx}..."):
            # Explain the batch
            all_cfs = ensemble.explain(batch_ds)

        # Group by original instance bytes for lookup
        grouped_cfs_map = cfs_group_by_original_data(all_cfs)
        
        quit_current_batch_early = False
        exit_app_flag = False

        # Iterate through each instance in the batch explicitly to maintain order and index
        for batch_i, instance in enumerate(batch_ds.data):
            global_idx = current_idx + batch_i
            instance_bytes = instance.tobytes()
            cfs = grouped_cfs_map.get(instance_bytes, [])
            
            if not cfs:
                # Should not happen usually if explainers are working, but possible
                continue

            # Identify which selectors select which CF
            cf_selection_map = {id(cf): set() for cf in cfs}
            
            for sel_name, selector in selectors.items():
                try:
                    selected_subset = selector(cfs)
                    for selected_cf in selected_subset:
                        cf_selection_map[id(selected_cf)].add(sel_name)
                except Exception as e:
                    console.print(f"[red]Selector {sel_name} failed: {e}[/red]")

            # Prepare display for this instance
            original_data = cfs[0].original_data
            # Original prediction
            orig_pred = model.predict(np.array([original_data]))[0]
            
            # Inverse transform for display
            orig_df = ds.inverse_transform(np.array([original_data]))
            
            picked_indices_session = set()

            # Interaction Loop for this instance
            while True:
                clear_screen()
                console.print(Panel(f"Instance Review ({global_idx + 1}/{total_instances})", style="bold blue")) # +1 for 1-based indexing for user
                console.print(f"Original Prediction: [bold cyan]{orig_pred}[/bold cyan]")
                console.print("Original Data:")
                console.print(orig_df.to_string(index=False))
                
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("#", style="dim", width=4)
                table.add_column("Features (Changes)") 
                table.add_column("Target")
                table.add_column("Explainer")
                table.add_column("Selected By")

                valid_indices = []
                for i, cf in enumerate(cfs):
                    valid_indices.append(i)
                    sel_by = list(cf_selection_map[id(cf)])
                    sel_str = ", ".join(sel_by) if sel_by else "[dim]-"
                    
                    diff_str = get_formatted_changes(ds, original_data, cf.data)
                    
                    # Style ID if picked
                    id_style = "bold green" if i in picked_indices_session else "dim"
                    
                    table.add_row(str(i), diff_str, str(cf.target_class), cf.explainer, sel_str, style=id_style)

                console.print(table)
                
                console.print("\n[b]Options:[/b] [green]p <#>[/green] or [green]p[/green] (all), [yellow]s[/yellow] to skip instance, [red]q[/red] to quit batch, [magenta]exit[/magenta] to exit app")
                choice = Prompt.ask("Action")
                
                if choice.lower() == 's':
                    break # Break out of interaction loop, continue to next instance in batch
                elif choice.lower() == 'q':
                    quit_current_batch_early = True
                    break # Break out of interaction loop
                elif choice.lower() == 'exit':
                    exit_app_flag = True
                    break # Break out of interaction loop
                elif choice.lower() == 'p':
                    # Pick all
                    for idx_to_pick in valid_indices:
                        if idx_to_pick not in picked_indices_session:
                             chosen_cf = cfs[idx_to_pick]
                             record = {
                                "original_data": chosen_cf.original_data.tolist(),
                                "cf_data": chosen_cf.data.tolist(),
                                "original_class": int(chosen_cf.original_class),
                                "target_class": int(chosen_cf.target_class),
                                "explainer": chosen_cf.explainer,
                                "selected_by": list(cf_selection_map[id(chosen_cf)]),
                                "timestamp": time.time()
                            }
                             save_result(record)
                             picked_indices_session.add(idx_to_pick)
                    console.print("[bold green]Saved all![/bold green]")
                    time.sleep(0.5)

                elif choice.lower().startswith('p '):
                    try:
                        idx_to_pick = int(choice.split()[1])
                        if idx_to_pick in valid_indices:
                            if idx_to_pick not in picked_indices_session:
                                chosen_cf = cfs[idx_to_pick]
                                record = {
                                    "original_data": chosen_cf.original_data.tolist(),
                                    "cf_data": chosen_cf.data.tolist(),
                                    "original_class": int(chosen_cf.original_class),
                                    "target_class": int(chosen_cf.target_class),
                                    "explainer": chosen_cf.explainer,
                                    "selected_by": list(cf_selection_map[id(chosen_cf)]),
                                    "timestamp": time.time()
                                }
                                save_result(record)
                                picked_indices_session.add(idx_to_pick)
                                console.print("[bold green]Saved![/bold green]")
                            else:
                                console.print("[yellow]Already picked[/yellow]")
                            time.sleep(0.5)
                        else:
                            console.print("[red]Invalid index[/red]")
                            time.sleep(1)
                    except (ValueError, IndexError):
                        console.print("[red]Invalid format. Use 'p <index>'[/red]")
                        time.sleep(1)
                else:
                     console.print("[red]Unknown command[/red]")
                     time.sleep(0.5)

            if exit_app_flag: # If exit was requested from within the instance review
                break # Break out of instance loop, then current_idx loop will check exit_app_flag
            if quit_current_batch_early:
                break # Break out of the batch_i loop as well

        current_idx = end_idx
        if exit_app_flag: # If exit was requested from any instance in this batch
            return
        if current_idx >= total_instances:
            console.print("[bold green]Finished all instances![/bold green]")
            break
            
        if not Confirm.ask("Continue to next batch?"):
            break

def main():
    while True:
        clear_screen()
        console.print(Panel("Titanic Counterfactuals TUI", style="bold red"))
        console.print("1. Start New Experiment (Pick CFs)")
        console.print("2. View Saved Results")
        console.print("3. Exit")
        
        choice = Prompt.ask("Select option", choices=["1", "2", "3"])
        
        if choice == "1":
            run_experiment()
        elif choice == "2":
            view_results()
        elif choice == "3":
            sys.exit(0)

if __name__ == "__main__":
    main()