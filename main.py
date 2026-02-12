from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import ast
import operator as op
import os
import functools
import inspect
from typing import List
from fastmcp import FastMCP
from mcp.types import Tool

# Define FastAPI app
app = FastAPI(title="Math API WEB MCP", description="API de operaciones matemáticas", version="1.0.0")

# ---------------------------
# Modelos Pydantic
# ---------------------------

class BinaryOperation(BaseModel):
    a: float
    b: float

class PowerOperation(BaseModel):
    base: float
    exponent: float

class RootOperation(BaseModel):
    n: float
    x: float

class ExpressionOperation(BaseModel):
    expr: str

# ---------------------------
# Decorador de logging
# ---------------------------

def api_log(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            print(f"ENDPOINT: {func.__name__} | RESULT: {result}")
            return result
        except Exception as e:
            print(f"ERROR in {func.__name__}: {e}")
            raise
    return wrapper

# ---------------------------
# Endpoints básicos
# ---------------------------

@app.post("/sumar", response_model=dict, tags=["Operaciones Básicas"])
@api_log
async def add(operation: BinaryOperation):
    """Suma dos números."""
    result = operation.a + operation.b
    return {"result": result, "operation": "sumar", "a": operation.a, "b": operation.b}

@app.post("/restar", response_model=dict, tags=["Operaciones Básicas"])
@api_log
async def sub(operation: BinaryOperation):
    """Resta b de a."""
    result = operation.a - operation.b
    return {"result": result, "operation": "restar", "a": operation.a, "b": operation.b}

@app.post("/multiplicar", response_model=dict, tags=["Operaciones Básicas"])
@api_log
async def mul(operation: BinaryOperation):
    """Multiplica dos números."""
    result = operation.a * operation.b
    return {"result": result, "operation": "multiplicar", "a": operation.a, "b": operation.b}

@app.post("/dividir", response_model=dict, tags=["Operaciones Básicas"])
@api_log
async def div(operation: BinaryOperation):
    """Divide a entre b."""
    if operation.b == 0:
        raise HTTPException(status_code=400, detail="Division by zero is not allowed.")
    result = operation.a / operation.b
    return {"result": result, "operation": "dividir", "a": operation.a, "b": operation.b}

@app.post("/modulo", response_model=dict, tags=["Operaciones Básicas"])
@api_log
async def mod(operation: BinaryOperation):
    """Calcula el resto de a / b."""
    result = operation.a % operation.b
    return {"result": result, "operation": "modulo", "a": operation.a, "b": operation.b}

# ---------------------------
# Operaciones avanzadas
# ---------------------------

@app.post("/potencia", response_model=dict, tags=["Operaciones Avanzadas"])
@api_log
async def power(operation: PowerOperation):
    """Eleva base a la potencia de exponent."""
    result = operation.base ** operation.exponent
    return {"result": result, "operation": "potencia", "base": operation.base, "exponent": operation.exponent}

@app.post("/raiz", response_model=dict, tags=["Operaciones Avanzadas"])
@api_log
async def root(operation: RootOperation):
    """Calcula la raíz n-ésima de x."""
    if operation.n == 0:
        raise HTTPException(status_code=400, detail="Cannot calculate 0-th root.")
    result = operation.x ** (1 / operation.n)
    return {"result": result, "operation": "raiz", "n": operation.n, "x": operation.x}

# ---------------------------
# Evaluador de expresiones
# ---------------------------

_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("Division by zero is not allowed.")
        return float(_ALLOWED_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        operand = _eval_node(node.operand)
        return float(_ALLOWED_UNARYOPS[type(node.op)](operand))
    raise ValueError("Unsupported expression.")

@app.post("/evaluar", response_model=dict, tags=["Operaciones Avanzadas"])
@api_log
async def eval_expr(operation: ExpressionOperation):
    """Evalúa una expresión matemática de forma segura."""
    expr = operation.expr.strip()
    if not expr:
        raise HTTPException(status_code=400, detail="Empty expression.")
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree)
        return {"result": result, "operation": "evaluar", "expr": operation.expr}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid expression: {str(e)}")

@app.get("/", tags=["Info"])
async def root():
    """Información de la API."""
    return {"name": "Math API", "version": "1.0.0", "docs": "/docs"}

# ---------------------------
# MCP Server Wrapper
# ---------------------------

mcp = FastMCP.from_fastapi(app)

# ---------------------------
# Explicit MCP Standard Method
# ---------------------------

@mcp.list_tools()
async def list_tools() -> List[Tool]:
    """
    Standard MCP tools/list method.
    """
    return [
        Tool(
            name="sumar",
            description="Suma dos números",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="restar",
            description="Resta b de a",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="multiplicar",
            description="Multiplica dos números",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="dividir",
            description="Divide a entre b",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="modulo",
            description="Calcula el módulo de a / b",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="potencia",
            description="Eleva base a la potencia exponent",
            inputSchema={
                "type": "object",
                "properties": {
                    "base": {"type": "number"},
                    "exponent": {"type": "number"}
                },
                "required": ["base", "exponent"]
            }
        ),
        Tool(
            name="raiz",
            description="Calcula la raíz n-ésima de x",
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {"type": "number"},
                    "x": {"type": "number"}
                },
                "required": ["n", "x"]
            }
        ),
        Tool(
            name="evaluar",
            description="Evalúa una expresión matemática segura",
            inputSchema={
                "type": "object",
                "properties": {
                    "expr": {"type": "string"}
                },
                "required": ["expr"]
            }
        )
    ]

# ---------------------------
# Run Server
# ---------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    mcp.run(transport="http", host="0.0.0.0", port=port)
