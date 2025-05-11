import tkinter as tk
from tkinter import ttk, StringVar, messagebox
import ttkbootstrap as ttb
from ttkbootstrap.constants import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import sympy as sp
import numpy as np
from sympy import symbols, diff, integrate, sympify, E, pi, oo, exp, sin, cos, tan, log, sqrt
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

class CalculusFunctionGrapher:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculus Function Grapher")
        self.root.geometry("1200x700")
        
        # Style configuration
        self.style = ttb.Style(theme="darkly")
        
        # Create the main frame
        self.main_frame = ttb.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Split the window into left and right panels
        self.left_panel = ttb.Frame(self.main_frame, bootstyle=SECONDARY, width= 300)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)
        
        self.right_panel = ttb.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Variables
        self.function_str = StringVar(value="e^(x^3)")
        self.x_min = tk.DoubleVar(value=-10)
        self.x_max = tk.DoubleVar(value=10)
        self.dark_mode = tk.BooleanVar(value=True)
        self.glass_mode = tk.BooleanVar(value=True)  # Variable for glassmorphic style
        self.selected_operations = {
            "original": tk.BooleanVar(value=True),
            "first_derivative": tk.BooleanVar(value=False),
            "second_derivative": tk.BooleanVar(value=False),
            "third_derivative": tk.BooleanVar(value=False),
            "integral": tk.BooleanVar(value=False)
        }
        
        # Colors for different modes - FIX: Use tuples for RGBA values instead of CSS-style strings
        self.colors = {
            "dark": {
                "bg": "#151a36",
                "plot_bg": "#1b2044",
                "grid": "#2a325e",
                "text": "white",
                "functions": ["#3584e4", "#ed333b", "#2ec27e", "#f5c211", "#9141ac"]
            },
            "light": {
                "bg": "#f0f0f0",
                "plot_bg": "#ffffff",
                "grid": "#cccccc",
                "text": "black",
                "functions": ["#1c71d8", "#c01c28", "#26a269", "#e5a50a", "#613583"]
            },
            "glass_dark": {
                "bg": "#151a36",
                "plot_bg": (0.106, 0.125, 0.267, 0.65),  # FIX: Use tuple instead of rgba string
                "grid": "#2a325e",
                "text": "white",
                "functions": ["#5294ff", "#ff5a63", "#4aeea0", "#ffe04a", "#c566ff"]
            },
            "glass_light": {
                "bg": "#f0f0f0",
                "plot_bg": (1, 1, 1, 0.65),  # FIX: Use tuple instead of rgba string
                "grid": "#cccccc",
                "text": "black",
                "functions": ["#4a8bef", "#ff4a59", "#3dde8a", "#ffdb3d", "#b14aff"]
            }
        }
        
        self.function_history = []
        self.current_theme_colors = self.colors["glass_dark"]  # Default to glass dark
        
        # Setup UI components
        self.setup_left_panel()
        self.setup_right_panel()
        
        # Initialize the plot
        self.setup_plot()
        
        # Add keyboard bindings
        self.root.bind("<Return>", lambda event: self.calculate_and_plot())
        
        # Apply glassmorphic style
        self.apply_glassmorphic_style()
        
    def apply_glassmorphic_style(self):
        """Apply glassmorphic styling to the application components"""
        bg_color = self.current_theme_colors["bg"]
        text_color = self.current_theme_colors["text"]
        
        # Apply styles to all frames if glass mode is enabled
        if self.glass_mode.get():
            # Use a more transparent background for panels
            for panel in [self.left_panel, self.right_panel]:
                panel.configure(style="Glass.TFrame")
                
            # Create custom styles for glassmorphic effect
            self.style.configure(
                "Glass.TFrame",
                background=bg_color,
                borderwidth=0,
                relief="raised",
                bordercolor=bg_color,
            )
            
            self.style.configure(
                "Glass.TButton",
                background=bg_color,
                foreground=text_color,
                borderwidth=0,
                focusthickness=0,
                highlightthickness=0
            )
            
            self.style.configure(
                "Glass.TEntry",
                fieldbackground=bg_color,
                foreground=text_color,
                borderwidth=0
            )
            
            self.style.configure(
                "Glass.TCheckbutton",
                background=bg_color,
                foreground=text_color
            )
            
        # Update plot styles for glassmorphic effect
        self.update_plot_styles()
        
    def setup_plot(self):
        # Initialize the plot
        self.fig = plt.figure(figsize=(6, 6), facecolor=self.current_theme_colors["bg"])
        self.ax = self.fig.add_subplot(111)
        
        # FIX: Handle plot background color based on type
        if isinstance(self.current_theme_colors["plot_bg"], tuple):
            # For RGBA tuple, convert to proper format
            plot_bg = self.current_theme_colors["plot_bg"]
        else:
            # For hex string, use directly
            plot_bg = self.current_theme_colors["plot_bg"]
            
        self.ax.set_facecolor(plot_bg)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar_frame = ttb.Frame(self.right_panel)
        self.toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        
        # Fix for the activebackground error - manually set compatible options
        for button in self.toolbar.winfo_children():
            if hasattr(button, 'config') and isinstance(button, tk.Button):
                button.config(background=self.current_theme_colors["bg"])
                # Don't set activebackground as it causes the error
        
        # Add an initial message to the plot
        self.ax.text(0.5, 0.5, "Results will appear here...", 
                    ha='center', va='center', color=self.current_theme_colors["text"], fontsize=12,
                    transform=self.ax.transAxes)
        
        # Set up grid and styles
        self.update_plot_styles()
        
        # Refresh the canvas
        self.canvas.draw()
        
    def update_plot_styles(self):
        # Apply glassmorphic effect to plot if enabled
        if self.glass_mode.get():
            if self.dark_mode.get():
                self.current_theme_colors = self.colors["glass_dark"]
            else:
                self.current_theme_colors = self.colors["glass_light"]
            
            # Create semi-transparent effect for plot background
            self.fig.patch.set_alpha(0.85)
            if hasattr(self.ax, 'patch'):
                self.ax.patch.set_alpha(0.7)
        else:
            if self.dark_mode.get():
                self.current_theme_colors = self.colors["dark"]
            else:
                self.current_theme_colors = self.colors["light"]
            
            # Reset transparency
            self.fig.patch.set_alpha(1.0)
            if hasattr(self.ax, 'patch'):
                self.ax.patch.set_alpha(1.0)
                
        # Apply styles to plot
        self.ax.grid(True, color=self.current_theme_colors["grid"], linestyle='--', alpha=0.7)
        self.ax.spines['bottom'].set_color(self.current_theme_colors["grid"])
        self.ax.spines['top'].set_color(self.current_theme_colors["grid"])
        self.ax.spines['left'].set_color(self.current_theme_colors["grid"])
        self.ax.spines['right'].set_color(self.current_theme_colors["grid"])
        self.ax.tick_params(colors=self.current_theme_colors["text"])
        self.ax.set_xlabel('x', color=self.current_theme_colors["text"])
        self.ax.set_ylabel('y', color=self.current_theme_colors["text"])
        self.fig.set_facecolor(self.current_theme_colors["bg"])
        
        # FIX: Handle plot background color based on type
        if isinstance(self.current_theme_colors["plot_bg"], tuple):
            # For RGBA tuple, we need to convert to proper format (0-1 scale)
            plot_bg = self.current_theme_colors["plot_bg"]
        else:
            # For hex string, use directly
            plot_bg = self.current_theme_colors["plot_bg"]
            
        self.ax.set_facecolor(plot_bg)
    
    def setup_left_panel(self):
        # Add glassmorphic panel effect
        left_frame = ttb.Frame(self.left_panel, bootstyle=SECONDARY)
        left_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title with glassmorphic effect
        title_label = ttb.Label(left_frame, text="Calculus Function Grapher", 
                               font=("Arial", 16, "bold"), bootstyle=LIGHT)
        title_label.pack(pady=(0, 10))
        
        # Subtitle
        subtitle_label = ttb.Label(left_frame, text="Visualize functions, derivatives, and integrals", 
                                 font=("Arial", 12))
        subtitle_label.pack(pady=(0, 20))
        
        # Function input
        function_frame = ttb.Frame(left_frame, bootstyle="dark")
        function_frame.pack(fill=tk.X, pady=(0, 20))
        
        function_label = ttb.Label(function_frame, text="f(x) =")
        function_label.pack(side=tk.LEFT, padx=(0, 10))
        
        function_entry = ttb.Entry(function_frame, textvariable=self.function_str, 
                                 bootstyle="dark")
        function_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        function_entry.bind("<Return>", lambda event: self.calculate_and_plot())
        function_entry.focus_set()
        
        # Calculator buttons
        button_frame = ttb.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Define calculator buttons
        button_config = [
            ['•', '^', '/', 'C', 'x'],
            ['sin', 'cos', 'tan', '(', ')'],
            ['log', '7', '8', '9', '/'],
            ['ln', '4', '5', '6', '*'],
            ['x²', '1', '2', '3', '-'],
            ['√', '-', '0', '←', '+'],
            ['e', 'π', 'abs', '|', '∞']
        ]
        
        # Create buttons with glassmorphic effect
        for row_idx, row in enumerate(button_config):
            row_frame = ttb.Frame(button_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            for col_idx, btn_text in enumerate(row):
                # Use a different bootstyle for special buttons to enhance the glassmorphic effect
                if btn_text in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    btn_style = "outline"
                else:
                    btn_style = "dark"
                    
                btn = ttb.Button(row_frame, text=btn_text, command=lambda t=btn_text: self.on_calculator_button(t),
                               width=4, bootstyle=btn_style)
                btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # X Range sliders with glassmorphic effect
        range_frame = ttb.Frame(left_frame)
        range_frame.pack(fill=tk.X, pady=(0, 20))
        
        x_range_label = ttb.Label(range_frame, text="X Range:")
        x_range_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Min and Max labels with entries for precise control
        min_max_frame = ttb.Frame(range_frame)
        min_max_frame.pack(fill=tk.X)
        
        min_label = ttb.Label(min_max_frame, text="Min:")
        min_label.pack(side=tk.LEFT, padx=(0, 5))
        
        min_entry = ttb.Entry(min_max_frame, textvariable=self.x_min, width=8)
        min_entry.pack(side=tk.LEFT, padx=(0, 10))
        min_entry.bind("<Return>", lambda event: self.calculate_and_plot())
        
        max_label = ttb.Label(min_max_frame, text="Max:")
        max_label.pack(side=tk.LEFT, padx=(10, 5))
        
        max_entry = ttb.Entry(min_max_frame, textvariable=self.x_max, width=8)
        max_entry.pack(side=tk.LEFT)
        max_entry.bind("<Return>", lambda event: self.calculate_and_plot())
        
        # Min slider with glassmorphic effect
        min_slider = ttb.Scale(range_frame, from_=-50, to=-1, variable=self.x_min, 
                            bootstyle="dark", length=200, 
                            command=lambda val: self.x_min.set(float(val)))
        min_slider.pack(fill=tk.X, pady=2)
        
        # Max slider with glassmorphic effect
        max_slider = ttb.Scale(range_frame, from_=1, to=50, variable=self.x_max, 
                            bootstyle="dark", length=200,
                            command=lambda val: self.x_max.set(float(val)))
        max_slider.pack(fill=tk.X)
        
        # Operations checkboxes with glassmorphic effect
        operations_frame = ttb.Labelframe(left_frame, text="Operations", bootstyle="light")
        operations_frame.pack(fill=tk.X, pady=(0, 20))
        
        operations = [
            ("Original Function", "original", self.current_theme_colors["functions"][0]),
            ("First Derivative", "first_derivative", self.current_theme_colors["functions"][1]),
            ("Second Derivative", "second_derivative", self.current_theme_colors["functions"][2]),
            ("Third Derivative", "third_derivative", self.current_theme_colors["functions"][3]),
            ("Integral", "integral", self.current_theme_colors["functions"][4])
        ]
        
        for text, key, color in operations:
            cb_frame = ttb.Frame(operations_frame)
            cb_frame.pack(fill=tk.X, padx=10, pady=2)
            
            cb = ttb.Checkbutton(cb_frame, text=text, variable=self.selected_operations[key], 
                                bootstyle="round-toggle")
            cb.pack(side=tk.LEFT, pady=2)
            
            # Add a color indicator with enhanced glow effect for glassmorphic style
            color_indicator = ttb.Label(cb_frame, text="■", foreground=color, font=("Arial", 12, "bold"))
            color_indicator.pack(side=tk.RIGHT)
        
        # Function history with glassmorphic effect
        history_frame = ttb.Labelframe(left_frame, text="Function History", bootstyle="light")
        history_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.history_combobox = ttb.Combobox(history_frame, bootstyle="dark")
        self.history_combobox.pack(fill=tk.X, padx=10, pady=10)
        self.history_combobox.set(self.function_str.get())
        self.history_combobox.bind("<<ComboboxSelected>>", self.on_history_selected)
        
        # Calculate button with glassmorphic effect
        calculate_btn = ttb.Button(left_frame, text="Calculate",
                                 command=self.calculate_and_plot, bootstyle="success")
        calculate_btn.pack(fill=tk.X, pady=5)
        
        # Style toggle buttons
        style_frame = ttb.Frame(left_frame)
        style_frame.pack(fill=tk.X, pady=5)
        
        # Dark mode toggle
        dark_mode_btn = ttb.Checkbutton(style_frame, text="Dark Mode",
                                      variable=self.dark_mode, 
                                      command=self.update_theme, 
                                      bootstyle="round-toggle")
        dark_mode_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Glass mode toggle - new feature
        glass_mode_btn = ttb.Checkbutton(style_frame, text="Glass Effect",
                                       variable=self.glass_mode, 
                                       command=self.update_theme, 
                                       bootstyle="round-toggle")
        glass_mode_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
    
    def setup_right_panel(self):
        # Function information panel at the bottom
        self.info_frame = ttb.Frame(self.right_panel)
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        # Labels for derivative and integral information with glassmorphic effect
        self.func_info_label = ttb.Label(self.info_frame, text="", font=("Arial", 9), anchor=tk.W, justify=tk.LEFT)
        self.func_info_label.pack(side=tk.LEFT, padx=5)
    
    def on_calculator_button(self, button_text):
        current_text = self.function_str.get()
        cursor_position = self.left_panel.focus_get().index(tk.INSERT) if hasattr(self.left_panel.focus_get(), 'index') else len(current_text)
        
        if button_text == 'C':
            # Clear
            self.function_str.set("")
        elif button_text == '←':
            # Backspace
            if cursor_position > 0:
                new_text = current_text[:cursor_position-1] + current_text[cursor_position:]
                self.function_str.set(new_text)
        elif button_text == 'x²':
            # Square
            new_text = current_text[:cursor_position] + "x^2" + current_text[cursor_position:]
            self.function_str.set(new_text)
        elif button_text == '√':
            # Square root
            new_text = current_text[:cursor_position] + "sqrt(" + current_text[cursor_position:]
            self.function_str.set(new_text)
        elif button_text == 'sin':
            new_text = current_text[:cursor_position] + "sin(" + current_text[cursor_position:]
            self.function_str.set(new_text)
        elif button_text == 'cos':
            new_text = current_text[:cursor_position] + "cos(" + current_text[cursor_position:]
            self.function_str.set(new_text)
        elif button_text == 'tan':
            new_text = current_text[:cursor_position] + "tan(" + current_text[cursor_position:]
            self.function_str.set(new_text)
        elif button_text == 'log':
            new_text = current_text[:cursor_position] + "log(" + current_text[cursor_position:]
            self.function_str.set(new_text)
        elif button_text == 'ln':
            new_text = current_text[:cursor_position] + "ln(" + current_text[cursor_position:]
            self.function_str.set(new_text)
        elif button_text == '•':
            new_text = current_text[:cursor_position] + "*" + current_text[cursor_position:]
            self.function_str.set(new_text)
        elif button_text == 'e':
            new_text = current_text[:cursor_position] + "e" + current_text[cursor_position:]
            self.function_str.set(new_text)
        elif button_text == 'π':
            new_text = current_text[:cursor_position] + "pi" + current_text[cursor_position:]
            self.function_str.set(new_text)
        elif button_text == 'abs':
            new_text = current_text[:cursor_position] + "abs(" + current_text[cursor_position:]
            self.function_str.set(new_text)
        elif button_text == '|':
            new_text = current_text[:cursor_position] + "|" + current_text[cursor_position:]
            self.function_str.set(new_text)
        elif button_text == '∞':
            new_text = current_text[:cursor_position] + "oo" + current_text[cursor_position:]
            self.function_str.set(new_text)
        else:
            # Add the button text at the cursor position
            new_text = current_text[:cursor_position] + button_text + current_text[cursor_position:]
            self.function_str.set(new_text)
    
    def on_history_selected(self, event):
        selected_function = self.history_combobox.get()
        self.function_str.set(selected_function)
        self.calculate_and_plot()
    
    def update_theme(self):
        """Update the theme based on dark mode and glass mode settings"""
        # Determine which theme to use
        if self.glass_mode.get():
            if self.dark_mode.get():
                self.current_theme_colors = self.colors["glass_dark"]
                self.style.theme_use("darkly")
            else:
                self.current_theme_colors = self.colors["glass_light"]
                self.style.theme_use("litera")
        else:
            if self.dark_mode.get():
                self.current_theme_colors = self.colors["dark"]
                self.style.theme_use("darkly")
            else:
                self.current_theme_colors = self.colors["light"]
                self.style.theme_use("litera")
        
        # Apply glassmorphic style if enabled
        self.apply_glassmorphic_style()
        
        # If there's a plot already, recalculate to update colors
        if hasattr(self, 'ax') and len(self.ax.lines) > 0:
            self.calculate_and_plot()
    
    def calculate_and_plot(self):
        try:
            # Clear the plot
            self.ax.clear()
            
            # Get the function string and x range
            func_str = self.function_str.get()
            x_min = self.x_min.get()
            x_max = self.x_max.get()
            
            # Validate x range
            if x_min >= x_max:
                messagebox.showerror("Invalid Range", "X min must be less than X max")
                return
            
            # Add to history if not already there
            if func_str not in self.function_history:
                self.function_history.append(func_str)
                self.history_combobox['values'] = self.function_history
            
            # Create symbolic representation
            x = sp.symbols('x')
            
            # Parse the function string
            # Replace some common syntax to make it more user-friendly
            parse_func_str = func_str.replace('^', '**')
            
            # Custom parsing to handle common math expressions
            if '|' in parse_func_str:
                # Handle absolute value syntax
                parts = parse_func_str.split('|')
                for i in range(1, len(parts), 2):
                    if i < len(parts):
                        abs_content = parts[i]
                        parts[i] = f"abs({abs_content})"
                parse_func_str = ''.join(parts)
            
            # Replace ln with log for sympy compatibility
            parse_func_str = parse_func_str.replace('ln(', 'log(')
            
            # Convert to sympy expression with implicit multiplication
            transformations = standard_transformations + (implicit_multiplication_application,)
            try:
                expr = parse_expr(parse_func_str, transformations=transformations)
            except Exception as e:
                # Try direct sympy parsing as fallback
                expr = sympify(parse_func_str)
            
            # Information for display
            func_info = []
            
            # Create a numpy lambda function for efficient plotting
            f = sp.lambdify(x, expr, modules=['numpy', {'log': np.log, 'ln': np.log}])
            
            # Generate x values
            x_vals = np.linspace(x_min, x_max, 1000)
            
            # Calculate y values, handling potential errors
            try:
                y_vals = f(x_vals)
                # Remove infinities and NaNs
                mask = np.isfinite(y_vals)
                x_vals_clean = x_vals[mask]
                y_vals_clean = y_vals[mask]
                
                # Find reasonable y-axis limits to prevent extreme zooming
                if len(y_vals_clean) > 0:
                    y_range = np.percentile(y_vals_clean, [5, 95])
                    y_padding = (y_range[1] - y_range[0]) * 0.2
                    y_min = y_range[0] - y_padding
                    y_max = y_range[1] + y_padding
                    
                    # Set limits but only if they're reasonable
                    if np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max:
                        self.ax.set_ylim(y_min, y_max)
                
                # Plot original function if selected
                if self.selected_operations["original"].get():
                    self.ax.plot(x_vals_clean, y_vals_clean, label='f(x)', 
                              color=self.current_theme_colors["functions"][0],
                              linewidth=2, alpha=0.9)  # slightly transparent for glass effect
                    func_info.append(f"f(x) = {sp.pretty(expr)}")
                
                # Calculate and plot first derivative if selected
                if self.selected_operations["first_derivative"].get():
                    f_prime = sp.diff(expr, x)
                    f_prime_func = sp.lambdify(x, f_prime, modules=['numpy', {'log': np.log, 'ln': np.log}])
                    try:
                        y_prime = f_prime_func(x_vals)
                        mask = np.isfinite(y_prime)
                        self.ax.plot(x_vals[mask], y_prime[mask], label="f'(x)", 
                                  color=self.current_theme_colors["functions"][1],
                                  linewidth=2, alpha=0.9)
                        func_info.append(f"f'(x) = {sp.pretty(f_prime)}")
                    except Exception as e:
                        func_info.append(f"f'(x) = {sp.pretty(f_prime)} (Error plotting: {str(e)})")
                
                # Calculate and plot second derivative if selected
                if self.selected_operations["second_derivative"].get():
                    f_double_prime = sp.diff(expr, x, 2)
                    f_double_prime_func = sp.lambdify(x, f_double_prime, modules=['numpy', {'log': np.log, 'ln': np.log}])
                    try:
                        y_double_prime = f_double_prime_func(x_vals)
                        mask = np.isfinite(y_double_prime)
                        self.ax.plot(x_vals[mask], y_double_prime[mask], label="f''(x)", 
                                  color=self.current_theme_colors["functions"][2],
                                  linewidth=2, alpha=0.9)
                        func_info.append(f"f''(x) = {sp.pretty(f_double_prime)}")
                    except Exception as e:
                        func_info.append(f"f''(x) = {sp.pretty(f_double_prime)} (Error plotting: {str(e)})")
                
                # Calculate and plot third derivative if selected
                if self.selected_operations["third_derivative"].get():
                    f_triple_prime = sp.diff(expr, x, 3)
                    f_triple_prime_func = sp.lambdify(x, f_triple_prime, modules=['numpy', {'log': np.log, 'ln': np.log}])
                    try:
                        y_triple_prime = f_triple_prime_func(x_vals)
                        mask = np.isfinite(y_triple_prime)
                        self.ax.plot(x_vals[mask], y_triple_prime[mask], label="f'''(x)", 
                                  color=self.current_theme_colors["functions"][3],
                                  linewidth=2, alpha=0.9)
                        func_info.append(f"f'''(x) = {sp.pretty(f_triple_prime)}")
                    except Exception as e:                        func_info.append(f"f'''(x) = {sp.pretty(f_triple_prime)} (Error plotting: {str(e)})")
                
                # Calculate and plot integral if selected
                if self.selected_operations["integral"].get():
                    f_integral = sp.integrate(expr, x)
                    f_integral_func = sp.lambdify(x, f_integral, modules=['numpy', {'log': np.log, 'ln': np.log}])
                    try:
                        y_integral = f_integral_func(x_vals)
                        mask = np.isfinite(y_integral)
                        self.ax.plot(x_vals[mask], y_integral[mask], label="∫f(x)dx", 
                                     color=self.current_theme_colors["functions"][4],
                                     linewidth=2, alpha=0.9)
                        func_info.append(f"∫f(x)dx = {sp.pretty(f_integral)}")
                    except Exception as e:
                        func_info.append(f"∫f(x)dx = {sp.pretty(f_integral)} (Error plotting: {str(e)})")
                
                # Add legend and grid
                self.ax.legend(loc="upper right", fontsize=10, frameon=False)
                self.ax.grid(True, color=self.current_theme_colors["grid"], linestyle='--', alpha=0.7)
                
                # Update function information display
                self.func_info_label.config(text="\n".join(func_info))
                
                # Refresh the canvas
                self.canvas.draw()
            
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while plotting: {str(e)}")
        
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    root = ttb.Window(themename="darkly")
    app = CalculusFunctionGrapher(root)
    root.mainloop()