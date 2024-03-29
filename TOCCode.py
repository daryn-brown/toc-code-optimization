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
    ('QUOTE',    r'\''),                  # Single quote
    ('DOUBLE_QUOTE', r'\"'),              # Double quote
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
    ('IDENTIFIER',       r'[A-Za-z_]\w*'),        # Identifiers
    ('STRING',   r'\".*?\"|\'.*?\''),     # String literals
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
                statements.append(self.parse_assignment_statement())
            else:
                self.next_token()  # Skip unrecognized tokens
        return statements

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
        value = self.current_token[1]
        self.match('NUMBER')
        self.match('END')
        return {'type': 'assignment', 'variable': var_name, 'value': value}

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

     


tabView = customtkinter.CTkTabview(app, width=400, height=500, corner_radius=0)
tabView.pack(padx=20, pady=20, fill="both", expand=True)

tabView.add("Tokenize")
tabView.add("Parse")

tabView.set("Tokenize")

textbox = customtkinter.CTkTextbox(tabView.tab("Tokenize"), width=400, height=200)
textbox.place(relx=0.5, rely=0.3, anchor="center")

tokens_textbox = customtkinter.CTkTextbox(tabView.tab("Tokenize"), width=400, height=200, state="disabled")
tokens_textbox.place(relx=0.5, rely=0.7, anchor="center")

parser_textbox = customtkinter.CTkTextbox(tabView.tab("Parse"), width=400, height=200, state="disabled")
parser_textbox.place(relx=0.5, rely=0.3, anchor="center")

tokenButton = customtkinter.CTkButton(tabView.tab("Tokenize"), text="Tokenize", command=print_tokens)
tokenButton.place(relx=0.5, rely=0.9, anchor="center")



app.mainloop()
