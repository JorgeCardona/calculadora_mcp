
# main.py
# API REST + MCP
# REST: https://calculadora-mcp.onrender.com/docs
# MCP:  https://calculadora-mcp.onrender.com/mcp

import ast
import functools
import inspect
import json
import math
import operator as op
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from mcp.types import Annotations, TextContent, ToolAnnotations


# =========================================================
# CONFIGURACIÓN
# =========================================================

VERSION = "2.1.3"
SERVICE_NAME = "calculadora-mcp"

mcp = FastMCP("Calculadora")


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
        result = float(
            _ALLOWED_UNARYOPS[type(node.op)](
                _eval_node(node.operand)
            )
        )

        if not math.isfinite(result):
            raise ValueError("El resultado no es finito.")

        return result

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

    except (
        SyntaxError,
        TypeError,
        OverflowError,
        ZeroDivisionError,
    ) as e:
        raise ValueError(f"Expresión inválida: {e}") from e


# =========================================================
# ANOTACIONES Y METADATOS MCP
# =========================================================

def anotaciones_calculo(title: str) -> ToolAnnotations:
    return ToolAnnotations(
        title=title,
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )


def metadatos_herramienta() -> dict:
    return {
        "service": SERVICE_NAME,
        "version": VERSION,
        "category": "matematicas",
    }


def crear_resultado_mcp(data: dict) -> ToolResult:
    operation = data.get("operation", "desconocida")

    # Anotaciones del contenido de la respuesta.
    # La audiencia "user" indica que el contenido está
    # destinado al usuario.
    content_annotations = Annotations(
        audience=["user"],
        priority=1.0,
    )

    # Metadatos personalizados del contenido.
    content_meta = {
        "service": SERVICE_NAME,
        "version": VERSION,
        "operation": operation,
        "content_format": "application/json",
    }

    # Metadatos personalizados de la ejecución.
    execution_meta = {
        "service": SERVICE_NAME,
        "version": VERSION,
        "operation": operation,
        "status": "success",
    }

    content = TextContent(
        type="text",
        text=json.dumps(data, ensure_ascii=False),
        annotations=content_annotations,
        meta=content_meta,
    )

    return ToolResult(
        content=[content],
        structured_content=data,
        meta=execution_meta,
    )


# =========================================================
# HERRAMIENTAS MCP
# Los nombres públicos se definen explícitamente.
# =========================================================

@mcp.tool(
    name="sumar",
    description="Suma dos números y devuelve el resultado.",
    annotations=anotaciones_calculo("Sumar"),
    meta=metadatos_herramienta(),
)
def tool_sumar(a: float, b: float) -> ToolResult:
    """Suma dos números."""
    return crear_resultado_mcp(calcular_suma(a, b))


@mcp.tool(
    name="restar",
    description="Resta b de a.",
    annotations=anotaciones_calculo("Restar"),
    meta=metadatos_herramienta(),
)
def tool_restar(a: float, b: float) -> ToolResult:
    """Resta b de a."""
    return crear_resultado_mcp(calcular_resta(a, b))


@mcp.tool(
    name="multiplicar",
    description="Multiplica dos números.",
    annotations=anotaciones_calculo("Multiplicar"),
    meta=metadatos_herramienta(),
)
def tool_multiplicar(a: float, b: float) -> ToolResult:
    """Multiplica dos números."""
    return crear_resultado_mcp(calcular_multiplicacion(a, b))


@mcp.tool(
    name="dividir",
    description="Divide a entre b.",
    annotations=anotaciones_calculo("Dividir"),
    meta=metadatos_herramienta(),
)
def tool_dividir(a: float, b: float) -> ToolResult:
    """Divide a entre b."""
    return crear_resultado_mcp(calcular_division(a, b))


@mcp.tool(
    name="modulo",
    description="Calcula el resto de a dividido entre b.",
    annotations=anotaciones_calculo("Módulo"),
    meta=metadatos_herramienta(),
)
def tool_modulo(a: float, b: float) -> ToolResult:
    """Calcula el resto de a dividido entre b."""
    return crear_resultado_mcp(calcular_modulo(a, b))


@mcp.tool(
    name="potencia",
    description="Eleva una base a un exponente.",
    annotations=anotaciones_calculo("Potencia"),
    meta=metadatos_herramienta(),
)
def tool_potencia(base: float, exponent: float) -> ToolResult:
    """Eleva base a la potencia exponent."""
    return crear_resultado_mcp(
        calcular_potencia(base, exponent)
    )


@mcp.tool(
    name="raiz",
    description="Calcula la raíz n-ésima de un número real.",
    annotations=anotaciones_calculo("Raíz"),
    meta=metadatos_herramienta(),
)
def tool_raiz(n: float, x: float) -> ToolResult:
    """Calcula la raíz n-ésima de x."""
    return crear_resultado_mcp(calcular_raiz(n, x))


@mcp.tool(
    name="evaluar",
    description="Evalúa una expresión matemática de forma segura.",
    annotations=anotaciones_calculo("Evaluar expresión"),
    meta=metadatos_herramienta(),
)
def tool_evaluar(expr: str) -> ToolResult:
    """Evalúa una expresión matemática de forma segura."""
    return crear_resultado_mcp(calcular_expresion(expr))


# =========================================================
# FASTAPI + INTEGRACIÓN HTTP DE MCP
# =========================================================

mcp_app = mcp.http_app(path="/")

app = FastAPI(
    title="Math API WEB MCP",
    description="Microservicio matemático con API REST y MCP",
    version=VERSION,
    lifespan=mcp_app.lifespan,
)


# =========================================================
# DECORADOR DE LOGGING
# =========================================================

def api_log(func):
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                print(
                    f"ENDPOINT: {func.__name__} | RESULT: {result}"
                )
                return result
            except Exception as e:
                print(f"ERROR en {func.__name__}: {e}")
                raise

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            print(
                f"ENDPOINT: {func.__name__} | RESULT: {result}"
            )
            return result
        except Exception as e:
            print(f"ERROR en {func.__name__}: {e}")
            raise

    return sync_wrapper


# =========================================================
# ENDPOINTS REST
# =========================================================

@app.post(
    "/sumar",
    tags=["Operaciones Básicas"],
    operation_id="rest_sumar",
)
@api_log
def rest_sumar(operation: BinaryOperation):
    return calcular_suma(operation.a, operation.b)


@app.post(
    "/restar",
    tags=["Operaciones Básicas"],
    operation_id="rest_restar",
)
@api_log
def rest_restar(operation: BinaryOperation):
    return calcular_resta(operation.a, operation.b)


@app.post(
    "/multiplicar",
    tags=["Operaciones Básicas"],
    operation_id="rest_multiplicar",
)
@api_log
def rest_multiplicar(operation: BinaryOperation):
    return calcular_multiplicacion(operation.a, operation.b)


@app.post(
    "/dividir",
    tags=["Operaciones Básicas"],
    operation_id="rest_dividir",
)
@api_log
def rest_dividir(operation: BinaryOperation):
    try:
        return calcular_division(operation.a, operation.b)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post(
    "/modulo",
    tags=["Operaciones Básicas"],
    operation_id="rest_modulo",
)
@api_log
def rest_modulo(operation: BinaryOperation):
    try:
        return calcular_modulo(operation.a, operation.b)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post(
    "/potencia",
    tags=["Operaciones Avanzadas"],
    operation_id="rest_potencia",
)
@api_log
def rest_potencia(operation: PowerOperation):
    try:
        return calcular_potencia(
            operation.base,
            operation.exponent,
        )
    except (ValueError, OverflowError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post(
    "/raiz",
    tags=["Operaciones Avanzadas"],
    operation_id="rest_raiz",
)
@api_log
def rest_raiz(operation: RootOperation):
    try:
        return calcular_raiz(operation.n, operation.x)
    except (ValueError, OverflowError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post(
    "/evaluar",
    tags=["Operaciones Avanzadas"],
    operation_id="rest_evaluar",
)
@api_log
def rest_evaluar(operation: ExpressionOperation):
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
        "version": VERSION,
        "docs": "/docs",
        "openapi": "/openapi.json",
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
