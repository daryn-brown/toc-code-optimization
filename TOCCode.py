import customtkinter
import re

customtkinter.set_appearance_mode("system")
customtkinter.set_default_color_theme("green")

app = customtkinter.CTk()

app.title("Code Optimizer")
app.geometry("800x600")

# Define our token types and regex patterns
token_specs = [
    ('NUMBER',   r'\d+(\.\d*)?'),         # Integer or decimal number
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

# A generator function that yields matched tokens
def tokenize(code):
    print("Tokenizing code...")  # Debugging print statement
    line_number = 1
    current_position = line_number_start = 0
    match = get_token(code)
    while match is not None:
        token_type = match.lastgroup
        token_value = match.group(token_type)
        if token_type == 'NEWLINE':
            line_number += 1
            line_number_start = current_position
        elif token_type == 'SKIP':
            pass
        elif token_type == 'MISMATCH':
            print(f'Warning: {line_number}:{current_position - line_number_start}: Illegal character {token_value!r}')
        elif token_type != 'WHITESPACE':  # Skip whitespace tokens
            if token_type == 'IDENTIFIER' and token_value in ['if', 'else', 'for', 'while']:
                token_type = 'KEYWORD'
            yield token_type, token_value
        current_position = match.end()
        match = get_token(code, current_position)
    if current_position != len(code):
        # Handle error: part of the code was not tokenized
        print(f'Warning: Unexpected end of input at {line_number}:{current_position - line_number_start}')

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
    parseButton = customtkinter.CTkButton(tabView.tab("Tokenize"), text="Parse", command=parse_code)
    parseButton.place(relx=0.5, rely=0.9, anchor="center")

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
        self.match('KEYWORD')  # Consume 'if'
        condition = self.parse_condition()
        self.match('END')  # Assuming a simplified syntax; consume ';'
        return {'type': 'if', 'condition': condition}

    def parse_while_statement(self):
        self.match('KEYWORD')  # Consume 'while'
        condition = self.parse_condition()
        self.match('END')  # Assuming a simplified syntax; consume ';'
        return {'type': 'while', 'condition': condition}

    def parse_assignment_statement(self):
        var_name = self.current_token[1]
        self.match('IDENTIFIER')
        self.match('ASSIGN')
        value = self.parse_expression()  # Parse the expression instead of directly expecting a number
        self.match('END')
        return {'type': 'assignment', 'variable': var_name, 'value': value}

    def parse_expression(self):
        # Parse a simple expression (number, identifier, or string literal)
        if self.current_token[0] in ['NUMBER', 'IDENTIFIER', 'STRING']:
            expression = self.current_token[1]
            self.next_token()
            return expression
        else:
            # Handle error
            expected_value = self.current_token[1] if self.current_token else 'EOF'
            print(f"Expected NUMBER, IDENTIFIER, or STRING, got {expected_value}")
            return None


    def parse_condition(self):
        # Simplified condition parsing
        left_operand = self.current_token[1]
        if self.current_token[0] == 'IDENTIFIER':
            self.next_token()
        else:
            print(f"Expected IDENTIFIER, got {self.current_token[1]}")
        
        operator = self.current_token[1]
        if operator in ['==', '!=', '>', '<', '>=', '<=']:
            self.next_token()  # Move to the next token
        else:
            print(f"Invalid comparison operator: {operator}")
        
        right_operand = self.current_token[1]
        if self.current_token[0] == 'NUMBER':
            self.next_token()  # Move to the next token
        else:
            print(f"Expected NUMBER, got {self.current_token[1]}")
        
        return {'left_operand': left_operand, 'operator': operator, 'right_operand': right_operand}
    
    def peek_next_token(self):
        # Temporarily save the state of the current token
        current = self.current_token
        # Get the next token
        self.next_token()
        next_token = self.current_token
        # Restore the state of the current token
        self.current_token = current
        return next_token
    
# Define a function to perform constant folding
def constant_folding(statements):
    # Traverse each statement in the syntax tree
    for statement in statements:
        if statement['type'] == 'assignment':
            # Check if the assignment statement involves constant expressions
            if all(isinstance(val, int) for val in [statement['value']]):
                # Perform constant folding by evaluating the expression
                result = evaluate_expression(statement['value'])
                # Replace the expression with the computed constant value
                statement['value'] = result
        elif statement['type'] == 'if':
            # Recursively apply constant folding to if statement conditions
            constant_folding([statement['condition']])
        elif statement['type'] == 'while':
            # Recursively apply constant folding to while statement conditions
            constant_folding([statement['condition']])

# Define a function to evaluate constant expressions
def evaluate_expression(expression):
    # Simply return the value of the constant expression
    return expression
# def print_folded():
#     # Replace the Tokenize button with a Parse button
    

# # Example usage
# if __name__ == "__main__":
#     # Sample syntax tree
#     statements = [
#         {'type': 'assignment', 'variable': 'x', 'value': 5},
#         {'type': 'assignment', 'variable': 'y', 'value': 10},
#         {'type': 'assignment', 'variable': 'z', 'value': 'x + y'}
#     ]

#     # Apply constant folding
#     constant_folding(statements)

#     # Print optimized statements
#     for statement in statements:
#         print(statement)
     


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

tokenButton = customtkinter.CTkButton(tabView.tab("Tokenize"), text="Tokenize", command=print_tokens)
tokenButton.place(relx=0.5, rely=0.9, anchor="center")

foldButton = customtkinter.CTkButton(tabView.tab("Parse"), text="Apply Constant Folding", command=constant_folding)
foldButton.place(relx=0.5, rely=0.7, anchor="center")



app.mainloop()
