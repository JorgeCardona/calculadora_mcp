
# main.py
# API REST + MCP
# REST: https://calculadora-mcp.onrender.com/docs
# MCP:  https://calculadora-mcp.onrender.com/mcp

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastmcp import FastMCP
import uvicorn
import ast
import operator as op
import os
import functools
import inspect
import math


# =========================================================
# MODELOS PYDANTIC
# =========================================================

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


# =========================================================
# LÓGICA DE NEGOCIO COMPARTIDA
# Estas funciones son utilizadas por REST y MCP.
# =========================================================

def calcular_suma(a: float, b: float) -> dict:
    return {
        "result": a + b,
        "operation": "sumar",
        "a": a,
        "b": b,
    }


def calcular_resta(a: float, b: float) -> dict:
    return {
        "result": a - b,
        "operation": "restar",
        "a": a,
        "b": b,
    }


def calcular_multiplicacion(a: float, b: float) -> dict:
    return {
        "result": a * b,
        "operation": "multiplicar",
        "a": a,
        "b": b,
    }


def calcular_division(a: float, b: float) -> dict:
    if b == 0:
        raise ValueError("No se puede dividir entre cero.")

    return {
        "result": a / b,
        "operation": "dividir",
        "a": a,
        "b": b,
    }


def calcular_modulo(a: float, b: float) -> dict:
    if b == 0:
        raise ValueError("No se puede calcular el módulo entre cero.")

    return {
        "result": a % b,
        "operation": "modulo",
        "a": a,
        "b": b,
    }


def calcular_potencia(base: float, exponent: float) -> dict:
    result = base ** exponent

    if isinstance(result, complex):
        raise ValueError("El resultado complejo no está soportado.")

    if not math.isfinite(float(result)):
        raise ValueError("El resultado no es un número finito.")

    return {
        "result": result,
        "operation": "potencia",
        "base": base,
        "exponent": exponent,
    }


def calcular_raiz(n: float, x: float) -> dict:
    if n == 0:
        raise ValueError("El índice de la raíz no puede ser cero.")

    if x < 0:
        if n.is_integer() and int(n) % 2 != 0:
            result = -((-x) ** (1 / n))
        else:
            raise ValueError(
                "Esta raíz no tiene resultado real para los valores dados."
            )
    else:
        result = x ** (1 / n)

    if not math.isfinite(float(result)):
        raise ValueError("El resultado no es un número finito.")

    return {
        "result": result,
        "operation": "raiz",
        "n": n,
        "x": x,
    }


# =========================================================
# EVALUADOR SEGURO DE EXPRESIONES
# =========================================================

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

    if (
        isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and isinstance(node.value, (int, float))
    ):
        return float(node.value)

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)

        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("División entre cero.")

        if isinstance(node.op, ast.Mod) and right == 0:
            raise ValueError("Módulo entre cero.")

        result = _ALLOWED_BINOPS[type(node.op)](left, right)

        if isinstance(result, complex):
            raise ValueError("No se admiten resultados complejos.")

        result = float(result)

        if not math.isfinite(result):
            raise ValueError("El resultado no es finito.")

        return result

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return float(_ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand)))

    raise ValueError("Expresión no permitida.")


def calcular_expresion(expr: str) -> dict:
    expr = expr.strip()

    if not expr:
        raise ValueError("La expresión está vacía.")

    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree)

        return {
            "result": result,
            "operation": "evaluar",
            "expr": expr,
        }

    except (SyntaxError, TypeError, OverflowError, ZeroDivisionError) as e:
        raise ValueError(f"Expresión inválida: {e}") from e


# =========================================================
# SERVIDOR MCP
# Los nombres de las herramientas se definen explícitamente.
# =========================================================

mcp = FastMCP("Calculadora")


@mcp.tool(name="sumar")
def tool_sumar(a: float, b: float) -> dict:
    """Suma dos números."""
    return calcular_suma(a, b)


@mcp.tool(name="restar")
def tool_restar(a: float, b: float) -> dict:
    """Resta b de a."""
    return calcular_resta(a, b)


@mcp.tool(name="multiplicar")
def tool_multiplicar(a: float, b: float) -> dict:
    """Multiplica dos números."""
    return calcular_multiplicacion(a, b)


@mcp.tool(name="dividir")
def tool_dividir(a: float, b: float) -> dict:
    """Divide a entre b."""
    return calcular_division(a, b)


@mcp.tool(name="modulo")
def tool_modulo(a: float, b: float) -> dict:
    """Calcula el resto de a dividido entre b."""
    return calcular_modulo(a, b)


@mcp.tool(name="potencia")
def tool_potencia(base: float, exponent: float) -> dict:
    """Eleva base a la potencia exponent."""
    return calcular_potencia(base, exponent)


@mcp.tool(name="raiz")
def tool_raiz(n: float, x: float) -> dict:
    """Calcula la raíz n-ésima de x."""
    return calcular_raiz(n, x)


@mcp.tool(name="evaluar")
def tool_evaluar(expr: str) -> dict:
    """Evalúa una expresión matemática de forma segura."""
    return calcular_expresion(expr)


# =========================================================
# INTEGRACIÓN HTTP DE MCP
# =========================================================

# MCP se monta en /mcp dentro de FastAPI.
# El path interno "/" permite que la URL final sea /mcp.
mcp_app = mcp.http_app(path="/")


# =========================================================
# APLICACIÓN FASTAPI
# FastAPI sirve REST y también aloja el servidor MCP.
# Se comparte el ciclo de vida de FastMCP para inicializar
# correctamente las sesiones del protocolo.
# =========================================================

app = FastAPI(
    title="Math API WEB MCP",
    description="Microservicio matemático con API REST y MCP",
    version="2.0.0",
    lifespan=mcp_app.lifespan,
)


# =========================================================
# DECORADOR DE LOGGING
# =========================================================

def api_log(func):
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            print(f"ENDPOINT: {func.__name__} | RESULT: {result}")
            return result

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"ENDPOINT: {func.__name__} | RESULT: {result}")
        return result

    return sync_wrapper


# =========================================================
# ENDPOINTS REST
# =========================================================

@app.post(
    "/sumar",
    response_model=dict,
    tags=["Operaciones Básicas"],
    operation_id="rest_sumar",
)
@api_log
def add(operation: BinaryOperation):
    return calcular_suma(operation.a, operation.b)


@app.post(
    "/restar",
    response_model=dict,
    tags=["Operaciones Básicas"],
    operation_id="rest_restar",
)
@api_log
def sub(operation: BinaryOperation):
    return calcular_resta(operation.a, operation.b)


@app.post(
    "/multiplicar",
    response_model=dict,
    tags=["Operaciones Básicas"],
    operation_id="rest_multiplicar",
)
@api_log
def mul(operation: BinaryOperation):
    return calcular_multiplicacion(operation.a, operation.b)


@app.post(
    "/dividir",
    response_model=dict,
    tags=["Operaciones Básicas"],
    operation_id="rest_dividir",
)
@api_log
def div(operation: BinaryOperation):
    try:
        return calcular_division(operation.a, operation.b)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post(
    "/modulo",
    response_model=dict,
    tags=["Operaciones Básicas"],
    operation_id="rest_modulo",
)
@api_log
def mod(operation: BinaryOperation):
    try:
        return calcular_modulo(operation.a, operation.b)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# =========================================================
# OPERACIONES AVANZADAS REST
# =========================================================

@app.post(
    "/potencia",
    response_model=dict,
    tags=["Operaciones Avanzadas"],
    operation_id="rest_potencia",
)
@api_log
def power(operation: PowerOperation):
    try:
        return calcular_potencia(operation.base, operation.exponent)
    except (ValueError, OverflowError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post(
    "/raiz",
    response_model=dict,
    tags=["Operaciones Avanzadas"],
    operation_id="rest_raiz",
)
@api_log
def root(operation: RootOperation):
    try:
        return calcular_raiz(operation.n, operation.x)
    except (ValueError, OverflowError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post(
    "/evaluar",
    response_model=dict,
    tags=["Operaciones Avanzadas"],
    operation_id="rest_evaluar",
)
@api_log
def eval_expr(operation: ExpressionOperation):
    try:
        return calcular_expresion(operation.expr)
    except (ValueError, OverflowError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# =========================================================
# ENDPOINT DE INFORMACIÓN
# =========================================================

@app.get("/", tags=["Info"])
def info():
    return {
        "name": "Math API WEB MCP",
        "version": "2.0.0",
        "docs": "/docs",
        "mcp": "/mcp",
        "tools": [
            "sumar",
            "restar",
            "multiplicar",
            "dividir",
            "modulo",
            "potencia",
            "raiz",
            "evaluar",
        ],
    }


# =========================================================
# MONTAR MCP EN FASTAPI
# =========================================================

app.mount("/mcp", mcp_app)


# =========================================================
# ARRANQUE LOCAL Y RENDER
# =========================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
