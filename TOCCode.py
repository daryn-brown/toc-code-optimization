import customtkinter
import re
import threading

customtkinter.set_appearance_mode("system")
customtkinter.set_default_color_theme("green")

app = customtkinter.CTk()

app.title("Code Optimizer")
app.geometry("800x600")

# Define our token types and regex patterns
token_specs = [
    ('NUMBER',   r'\d+(\.\d*)?'),         # Integer or decimal number
    ('KEYWORD',  r'\b(if|else|while|for|break)\b'), # Keywords
    ('EQUALS',   r'=='),                  # Equality operator
    ('ASSIGN',   r'='),                   # Assignment operator
    ('PLUS',     r'\+'),                  # Addition operator
    ('MINUS',    r'-'),                   # Subtraction operator
    ('MULTIPLY', r'\*'),                  # Multiplication operator
    ('DIVIDE',   r'/'),                   # Division operator
    ('MODULO',   r'%'),                   # Modulo operator
    ('POWER',    r'\*\*'),                # Exponentiation operator
    ('INCREMENT', r'\+\+'),               # Increment operator
    ('DECREMENT', r'--'),                 # Decrement operator
    ('GT',       r'>'),                   # Greater than operator
    ('LT',       r'<'),                   # Less than operator
    ('GTE',      r'>='),                  # Greater than or equal to operator
    ('LTE',      r'<='),                  # Less than or equal to operator
    ('NE',       r'!='),                  # Not equal to operator
    ('LPAREN',   r'\('),                  # Left parenthesis
    ('RPAREN',   r'\)'),                  # Right parenthesis
    ('LBRACE',   r'\{'),                  # Left brace
    ('RBRACE',   r'\}'),                  # Right brace
    ('LSQUARE',  r'\['),                  # Left square bracket
    ('RSQUARE',  r'\]'),                  # Right square bracket
    ('COMMA',    r','),                   # Comma
    ('PERIOD',   r'\.'),                  # Period (dot)
    ('COLON',    r':'),                   # Colon
    ('SEMICOLON', r';'),                  # Semicolon
    ('BACKSLASH', r'\\'),                 # Backslash
    ('PIPE',     r'\|'),                  # Pipe (bitwise or)
    ('AMPERSAND', r'&'),                  # Ampersand (bitwise and)
    ('CARET',    r'\^'),                  # Caret (bitwise xor)
    ('TILDE',    r'~'),                   # Tilde (bitwise not)
    ('BOOL_AND', r'and'),                 # Boolean and
    ('BOOL_OR',  r'or'),                  # Boolean or
    ('BOOL_NOT', r'not'),                 # Boolean not
    ('WHITESPACE', r'\s+'),               # Whitespace
    ('NEWLINE',  r'\n'),                  # Line endings
    ('STRING',   r'"(\\.|[^"\\])*"|\'(\\.|[^\'\\])*\''), # String literals with support for escaped quotes
    ('IDENTIFIER',       r'[A-Za-z_]\w*'),        # Identifiers
    ('QUOTE',    r'\''),                  # Single quote
    ('DOUBLE_QUOTE', r'\"'),              # Double quote
    ('COMMENT',  r'#.*'),                 # Comments
    ('SKIP',     r'[ \t]+'),              # Skip over spaces and tabs
    ('MISMATCH', r'.'),                   # Any other character
]

# Create a regex that matches any of the above token types
tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specs)

# Compile the regex
get_token = re.compile(tok_regex, re.DOTALL).match

indent_level = 0  # Current level of indentation
indent_stack = [0]  # Stack to keep track of indentation levels (0 represents the base level)


# A generator function that yields matched tokens
def tokenize(code):
    print("Tokenizing code...")
    line_number = 1
    current_position = line_number_start = 0
    global indent_level, indent_stack  # Use global variables for indentation tracking

    lines = code.split('\n')
    for line in lines:
        # Calculate the indentation level of the current line (number of leading spaces)
        new_indent_level = len(line) - len(line.lstrip(' '))
        if new_indent_level > indent_stack[-1]:
            # If the new indent level is greater, we have an INDENT
            indent_stack.append(new_indent_level)
            yield 'INDENT', new_indent_level
        while new_indent_level < indent_stack[-1]:
            # If the new indent level is lesser, we have a DEDENT for each level we go back
            indent_stack.pop()
            yield 'DEDENT', indent_stack[-1]
        
        # Tokenize the rest of the line as before
        current_position = 0
        match = get_token(line)
        while match is not None:
            token_type = match.lastgroup
            token_value = match.group(token_type)
            if token_type not in ['WHITESPACE', 'NEWLINE', 'SKIP']:  # Handle tokens as before, but ignore leading whitespace
                yield token_type, token_value
            current_position = match.end()
            match = get_token(line, current_position)

        yield 'NEWLINE', '\n'  # Yield a NEWLINE at the end of each line
        line_number += 1


def print_tokens():
    global tokens_textbox

    code = textbox.get("0.0", "end")
    tokens_textbox.configure(state="normal")  # Enable editing the textbox
    tokens_textbox.delete("0.0", "end")  # Clear the current content
    for token in tokenize(code):
        tokens_textbox.insert("end", f"{token}\n")  # Insert the token into the textbox
    tokens_textbox.configure(state="disabled")  # Disable editing the textbox
    # Replace the Tokenize button with a Parse button
    tokenButton.destroy()  # Remove the Tokenize button
    parseButton = customtkinter.CTkButton(tabView.tab("Tokenize"), text="Parse", command=parse_code_thread)
    parseButton.place(relx=0.5, rely=0.9, anchor="center")

# Define a function to parse the code in a separate thread
def parse_code_thread():
    # Call the parse_code function inside a new thread
    threading.Thread(target=parse_code).start()


def parse_code():
    global parser_textbox

    # Get the code from the textbox
    code = textbox.get("0.0", "end")

    # Tokenize the code
    tokens = tokenize(code)

    # Create a parser instance
    parser = Parser(tokens)

    # Parse the code
    parsed_statements = parser.parse()

    # Display parsed statements in the parser_textbox
    parser_textbox.configure(state="normal")  # Enable editing the textbox
    parser_textbox.delete("0.0", "end")  # Clear the current content
    for statement in parsed_statements:
        parser_textbox.insert("end", f"{statement}\n")  # Insert the parsed statement into the textbox
    parser_textbox.configure(state="disabled")  # Disable editing the textbox

    # Switch to the "Parse" tab
    tabView.set("Parse")



# Parser (Syntax Analyzer)
class Parser:
    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.current_token = None
        self.next_token()
    
    def next_token(self):
        try:
            self.current_token = next(self.tokens)
        except StopIteration:
            self.current_token = None
    
    def match(self, expected_type):
        if self.current_token and self.current_token[0] == expected_type:
            self.next_token()
        else:
            expected_value = self.current_token[1] if self.current_token else 'EOF'
            print(f"Expected {expected_type}, got {expected_value}")

    def parse(self):
        statements = []
        while self.current_token:
            if self.current_token[0] == 'KEYWORD':
                if self.current_token[1] == 'if':
                    statements.append(self.parse_if_statement())
                elif self.current_token[1] == 'while':
                    statements.append(self.parse_while_statement())
            elif self.current_token[0] == 'IDENTIFIER':
                # Peek at the next token to determine if it's a function call or assignment
                if self.peek_next_token()[0] == 'LPAREN':
                    statements.append(self.parse_function_call())
                else:
                    statements.append(self.parse_assignment_statement())
            else:
                self.next_token()  # Skip unrecognized tokens
        return statements
    
    def parse_function_call(self):
        function_name = self.current_token[1]
        self.match('IDENTIFIER')
        self.match('LPAREN')  # Consume the left parenthesis '('

        args = []
        if self.current_token[0] != 'RPAREN':
            args.append(self.parse_expression())  # Parse the first argument

        while self.current_token[0] == 'COMMA':
            self.match('COMMA')  # Consume the comma ','
            args.append(self.parse_expression())  # Parse the next argument

        self.match('RPAREN')  # Consume the right parenthesis ')'
        # Assuming function call must be followed by a semicolon ';'
        self.match('SEMICOLON')

        return {'type': 'function_call', 'name': function_name, 'arguments': args}

    def parse_if_statement(self):
        self.match('KEYWORD')  # Consumes 'if'
        condition = self.parse_expression()  # Parses the condition
        
        self.match('COLON')
        self.match('NEWLINE')
        self.match('INDENT')
        
        if_body = self.parse_block()  # Parses the 'if' body
        
        else_body = []
        # Check if there's an 'else' part following the 'if' part
        next_token = self.peek_next_token()
        if next_token and next_token[1] == 'else':
            self.next_token()  # Consumes 'else'
            self.match('COLON')
            self.match('NEWLINE')
            self.match('INDENT')
            else_body = self.parse_block()  # Parses the 'else' block
            self.match('DEDENT')  # It's crucial to match 'DEDENT' after parsing the 'else' block
        
        self.match('DEDENT')  # Matching 'DEDENT' after parsing the 'if' block, ensuring we exit the block correctly
        
        return {
            'type': 'if_statement',
            'condition': condition.rstrip(' :'),  # Stripping trailing spaces and colon for cleaner condition
            'if_body': if_body,
            'else_body': else_body
        }

    def parse_block(self):
        block_statements = []
        while self.current_token and self.current_token[0] != 'DEDENT':
            if self.current_token[0] == 'IDENTIFIER':
                if self.peek_next_token()[0] == 'LPAREN':
                    block_statements.append(self.parse_function_call())
                    self.next_token()  # Ensure to move past the function call
                else:
                    block_statements.append(self.parse_assignment_statement())
            elif self.current_token[0] == 'KEYWORD':
                if self.current_token[1] in ['if', 'while', 'for']:
                    block_statements.append(self.parse_if_statement() if self.current_token[1] == 'if' else self.parse_while_statement())
            else:
                self.next_token()  # Advance to the next token if the current one doesn't match expected types

            self.next_token()  # Ensure to move to the next token at the end of the loop

        # Correctly handle the DEDENT token signaling the end of the block
        if self.current_token and self.current_token[0] == 'DEDENT':
            self.next_token()  # Consume the DEDENT token to exit the block

        return block_statements




    def parse_while_statement(self):
        self.match('KEYWORD')  # Consume 'while'
        
        # Parse the condition directly here
        condition_parts = []
        while self.current_token and self.current_token[1] != ':':
            condition_parts.append(self.current_token[1])
            self.next_token()
        condition = ' '.join(condition_parts)

    def parse_assignment_statement(self):
        var_name = self.current_token[1]
        self.match('IDENTIFIER')
        self.match('ASSIGN')
        value = self.parse_expression()  # Parse the expression instead of directly expecting a number
        self.match('END')
        return {'type': 'assignment', 'variable': var_name, 'value': value}

    def parse_expression(self):
        expression_parts = []

        # Keep parsing until we reach a token that signifies the end of the expression,
        # including the newline character as one of those tokens.
        while self.current_token and self.current_token[0] not in ['SEMICOLON', 'RPAREN', 'COMMA', 'NEWLINE']:
            expression_parts.append(self.current_token[1])
            self.next_token()  # Move to the next token

        # Join the collected parts into a single string representing the full expression
        return ' '.join(expression_parts)
    
    def peek_next_token(self):
        # Temporarily save the state of the current token
        current = self.current_token
        try:
            self.next_token()  # Advance to the next token
            next_token = self.current_token  # Store the next token
        finally:
            # Restore the state of the current token
            self.current_token = current
        return next_token

    
    
# Define a function to perform constant folding
def constant_folding(statements):

    if statements is None:
        return []

    optimized_statements = []
    for statement in statements:
        if statement is None:
            continue
        if statement['type'] == 'assignment':
            # Attempt to optimize assignment statements as before
            try:
                if isinstance(statement['value'], str) and all(char.isdigit() or char in " +-*/()" for char in statement['value']):
                    statement['value'] = str(eval(statement['value']))
                optimized_statements.append(statement)
            except Exception as e:
                print(f"Skipping optimization due to error: {e}")
                optimized_statements.append(statement)
        elif statement['type'] == 'if_statement':
            # Apply constant folding to the bodies of the if and else blocks
            statement['if_body'] = constant_folding(statement['if_body'])
            statement['else_body'] = constant_folding(statement['else_body'])
            optimized_statements.append(statement)
        else:
            # Handle other statement types as necessary
            optimized_statements.append(statement)
    return optimized_statements



# Define a function to evaluate constant expressions
def evaluate_expression(expression):
    # If the expression is just a number (int or float), return it directly
    try:
        return eval(expression)
    except NameError:
        # If the expression involves variables or is not purely numeric, return it as is
        return expression

def display_folded_code():
    tabView.set("Constant Folding")
    code = textbox.get("1.0", "end-1c")
    tokens = list(tokenize(code))
    parser = Parser(tokens)
    parsed_statements = parser.parse()
    folded_statements = constant_folding(parsed_statements)

    folding_textbox.configure(state="normal")
    folding_textbox.delete("1.0", "end")

    for statement in folded_statements:
        if statement['type'] == 'assignment':
            folding_textbox.insert("end", f"{statement['variable']} = {statement['value']}\n")
        elif statement['type'] == 'function_call':
            args = ', '.join([str(arg) for arg in statement['arguments']])
            folding_textbox.insert("end", f"{statement['name']}({args})\n")
        elif statement['type'] == 'if':
            condition = " ".join([statement['condition'][part] for part in ['left_operand', 'operator', 'right_operand']])
            folding_textbox.insert("end", f"if {condition}:\n")
            # Note: You might need to handle the body of the if statement.
        # Handle other types of statements (e.g., 'while', 'for', etc.) as needed

    folding_textbox.configure(state="disabled")

def eliminate_dead_code(parsed_statements):
    # Identify used variables
    used_variables = set()
    for statement in parsed_statements:
        if statement['type'] in ['function_call', 'if_statement']:
            # This simplistic check assumes that all arguments and conditions involve variable usage
            # You might need a more sophisticated analysis depending on your syntax
            if statement['type'] == 'function_call':
                for arg in statement['arguments']:
                    used_variables.update(extract_variables(arg))
            elif statement['type'] == 'if_statement':
                used_variables.update(extract_variables(statement['condition']))
                used_variables.update(extract_variables_from_block(statement['if_body']))
                used_variables.update(extract_variables_from_block(statement['else_body']))
        elif statement['type'] == 'assignment':
            # Assuming right-hand side might involve variable usage
            used_variables.update(extract_variables(statement['value']))

    # Eliminate dead assignments
    optimized_statements = []
    for statement in parsed_statements:
        if statement['type'] == 'assignment' and statement['variable'] not in used_variables:
            # Skip adding this statement to optimized_statements
            continue
        optimized_statements.append(statement)

    return optimized_statements

def extract_variables(expression):
    # Dummy function to extract variables from an expression string
    # Implement based on your project's needs
    return set(re.findall(r'\b[A-Za-z_]\w*\b', expression))

def extract_variables_from_block(block):
    variables = set()
    for statement in block:
        if statement is not None:  # Add this check to skip None values
            if statement['type'] == 'assignment':
                variables.update(extract_variables(statement['value']))
            elif statement['type'] == 'function_call':
                for arg in statement['arguments']:
                    variables.update(extract_variables(arg))
    return variables


def display_eliminated_code():
    tabView.set("Dead Code Elimination")
    code = textbox.get("1.0", "end-1c")  # Get current code
    tokens = list(tokenize(code))  # Tokenize the code
    parser = Parser(tokens)  # Initialize the parser
    parsed_statements = parser.parse()  # Parse the code to get parsed statements
    
    eliminated_statements = eliminate_dead_code(parsed_statements)  # Apply dead code elimination
    
    # Prepare and display the optimized code
    deadCode_textbox.configure(state="normal")
    deadCode_textbox.delete("1.0", "end")  # Clear existing content
    for statement in eliminated_statements:
        if statement['type'] == 'assignment':
            deadCode_textbox.insert("end", f"{statement['variable']} = {statement['value']}\n")
        elif statement['type'] == 'function_call':
            args = ', '.join(statement['arguments'])
            deadCode_textbox.insert("end", f"{statement['name']}({args});\n")
        # Add more conditions as needed for other types of statements
    deadCode_textbox.configure(state="disabled")
    
    # Switch to the "Dead Code Elimination" tab
    tabView.set("Dead Code Elimination")

     


tabView = customtkinter.CTkTabview(app, width=400, height=500, corner_radius=0)
tabView.pack(padx=20, pady=20, fill="both", expand=True)

tabView.add("Tokenize")
tabView.add("Parse")
tabView.add("Constant Folding")
tabView.add("Dead Code Elimination")


tabView.set("Tokenize")

textbox = customtkinter.CTkTextbox(tabView.tab("Tokenize"), width=400, height=200)
textbox.place(relx=0.5, rely=0.3, anchor="center")

tokens_textbox = customtkinter.CTkTextbox(tabView.tab("Tokenize"), width=400, height=200, state="disabled")
tokens_textbox.place(relx=0.5, rely=0.7, anchor="center")

parser_textbox = customtkinter.CTkTextbox(tabView.tab("Parse"), width=400, height=200, state="disabled")
parser_textbox.place(relx=0.5, rely=0.3, anchor="center")

folding_textbox = customtkinter.CTkTextbox(tabView.tab("Constant Folding"), width=400, height=200, state="disabled")
folding_textbox.place(relx=0.5, rely=0.5, anchor="center")

deadCode_textbox = customtkinter.CTkTextbox(tabView.tab("Dead Code Elimination"), width=400, height=200, state="disabled")
deadCode_textbox.place(relx=0.5, rely=0.5, anchor="center")

tokenButton = customtkinter.CTkButton(tabView.tab("Tokenize"), text="Tokenize", command=print_tokens)
tokenButton.place(relx=0.5, rely=0.9, anchor="center")

foldButton = customtkinter.CTkButton(tabView.tab("Parse"), text="Apply Constant Folding", command=display_folded_code)
foldButton.place(relx=0.5, rely=0.7, anchor="center")

deadCodeButton = customtkinter.CTkButton(tabView.tab("Constant Folding"), text="Apply Dead Code Elimination", command=display_eliminated_code)
deadCodeButton.place(relx=0.5, rely=0.7, anchor="center")

app.mainloop()
