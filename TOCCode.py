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
        self.match('if')
        condition = self.parse_expression()  # Parse the condition
        self.match('COLON')  # Assuming a colon follows the condition in Python syntax
        body = self.parse_block()  # Parse the block of statements forming the body

        else_body = None
        if self.current_token and self.current_token[0] == 'KEYWORD' and self.current_token[1] == 'else':
            self.match('else')
            self.match('COLON')
            else_body = self.parse_block()  # Parse the else block

        return {
            'type': 'if',
            'condition': condition,
            'body': body,
            'else_body': else_body
        }
    
    def parse_block(self):
        # A method to parse a block of statements until the end of the block is reached.
        # This could be determined by a decrease in indentation level, or by reaching an 'else',
        # 'elif', 'except', 'finally', or similar.
        statements = []
        while not self.is_end_of_block():
            statement = self.parse_next_statement()
            if statement:
                statements.append(statement)
        return statements
    
    def is_end_of_block(self):
        # Check if the next token indicates a dedent, which would mean the end of the current block.
        # This assumes your tokenizer generates 'INDENT' and 'DEDENT' tokens to represent changes in indentation.
        # Additionally, check for any token that should logically terminate a block (e.g., 'else', 'elif', 'except', etc.),
        # but ensure to differentiate cases where these tokens might appear within the block itself (like in nested if-else structures).

        if not self.current_token:
            # End of file
            return True

        # Assuming you have 'DEDENT' tokens to mark the end of blocks
        if self.current_token[0] == 'DEDENT':
            return True

        # Assuming 'NEWLINE' tokens are used and might precede 'DEDENT' tokens or indicate a pause in statements.
        # However, 'NEWLINE' on its own isn't always an indication of the end of a block, as blocks might contain blank lines.
        if self.current_token[0] == 'NEWLINE':
            # Peek at the next token to see if it's 'DEDENT' or another 'NEWLINE'.
            next_token = self.peek_next_token()
            if next_token and next_token[0] == 'DEDENT':
                return True

        # Check for other tokens that might logically end a block, if applicable
        # This part depends on the structure of your language and parser

        return False


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
        expression_parts = []

        # Keep parsing until we reach a token that signifies the end of the expression,
        # including the newline character as one of those tokens.
        while self.current_token and self.current_token[0] not in ['SEMICOLON', 'RPAREN', 'COMMA', 'NEWLINE']:
            expression_parts.append(self.current_token[1])
            self.next_token()  # Move to the next token

        # Join the collected parts into a single string representing the full expression
        return ' '.join(expression_parts)



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
    # Modified to evaluate expressions where possible but retain all statements
    for statement in statements:
        if statement['type'] == 'assignment':
            # Attempt to evaluate the expression if it's purely numeric
            try:
                # Only attempt evaluation if it's safe (no variables/functions)
                if isinstance(statement['value'], str) and all(char.isdigit() or char in " +-*/()" for char in statement['value']):
                    statement['value'] = str(eval(statement['value']))
            except Exception as e:
                print(f"Skipping optimization due to error: {e}")
    return statements

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
    # Get the code from the textbox
    code = textbox.get("1.0", "end-1c")

    # Tokenize and parse the code to get parsed statements
    tokens = list(tokenize(code))  # Ensure tokenize returns a list or convert it to a list
    parser = Parser(tokens)
    parsed_statements = parser.parse()

    # Apply constant folding to the parsed statements
    folded_statements = constant_folding(parsed_statements)

    # Display the folded code in the folding_textbox
    folding_textbox.configure(state="normal")  # Enable editing the textbox
    folding_textbox.delete("1.0", "end")  # Clear the current content

    # Assuming parsed_statements is a list of dictionaries as mentioned
    for statement in folded_statements:
        if 'variable' in statement and 'value' in statement:
            folding_textbox.insert("end", f"{statement['variable']} = {statement['value']}\n")

    folding_textbox.configure(state="disabled")  # Disable editing the textbox
    # Switch to the "Parse" tab
    

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

folding_textbox = customtkinter.CTkTextbox(tabView.tab("Constant Folding"), width=400, height=200, state="disabled")
folding_textbox.place(relx=0.5, rely=0.5, anchor="center")

tokenButton = customtkinter.CTkButton(tabView.tab("Tokenize"), text="Tokenize", command=print_tokens)
tokenButton.place(relx=0.5, rely=0.9, anchor="center")

foldButton = customtkinter.CTkButton(tabView.tab("Parse"), text="Apply Constant Folding", command=display_folded_code)
foldButton.place(relx=0.5, rely=0.7, anchor="center")



app.mainloop()
