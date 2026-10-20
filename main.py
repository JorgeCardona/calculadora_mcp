# main.py
# API REST + MCP
# REST: https://calculadora-mcp.onrender.com/docs
# MCP:  https://calculadora-mcp.onrender.com/mcp

import ast
import logging
import math
import operator as op
import os
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typeguard import typechecked
from fastmcp import FastMCP
from mcp.types import ToolAnnotations


# =========================================================
# CONFIGURACIÓN
# =========================================================

VERSION = "2.1.3"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("calculadora")


# =========================================================
# LOGGING: DIFERENCIA REST Y MCP
# =========================================================

def registrar_llamada(origen: str, nombre: str):
    def decorador(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(
                "[%s] INICIO | herramienta=%s",
                origen,
                nombre,
            )
            try:
                resultado = func(*args, **kwargs)
                logger.info(
                    "[%s] ÉXITO | herramienta=%s",
                    origen,
                    nombre,
                )
                return resultado
            except Exception:
                logger.exception(
                    "[%s] ERROR | herramienta=%s",
                    origen,
                    nombre,
                )
                raise

        return wrapper
    return decorador


# =========================================================
# MODELOS
# =========================================================

class BinaryOperation(BaseModel):
    model_config = ConfigDict(strict=True)
    a: float
    b: float


class PowerOperation(BaseModel):
    model_config = ConfigDict(strict=True)
    base: float
    exponent: float


class RootOperation(BaseModel):
    model_config = ConfigDict(strict=True)
    n: float
    x: float


class ExpressionOperation(BaseModel):
    model_config = ConfigDict(strict=True)
    expr: str = Field(min_length=1, max_length=500)


# =========================================================
# LÓGICA DE NEGOCIO COMPARTIDA
# =========================================================

@typechecked
def calcular_suma(a: float, b: float) -> dict:
    resultado = a + b
    if not math.isfinite(resultado):
        raise ValueError("El resultado no es finito.")
    return {
        "result": resultado,
        "operation": "sumar",
        "a": a,
        "b": b,
    }


@typechecked
def calcular_resta(a: float, b: float) -> dict:
    resultado = a - b
    if not math.isfinite(resultado):
        raise ValueError("El resultado no es finito.")
    return {
        "result": resultado,
        "operation": "restar",
        "a": a,
        "b": b,
    }


@typechecked
def calcular_multiplicacion(a: float, b: float) -> dict:
    resultado = a * b
    if not math.isfinite(resultado):
        raise ValueError("El resultado no es finito.")
    return {
        "result": resultado,
        "operation": "multiplicar",
        "a": a,
        "b": b,
    }


@typechecked
def calcular_division(a: float, b: float) -> dict:
    if b == 0:
        raise ValueError("No se puede dividir entre cero.")
    resultado = a / b
    if not math.isfinite(resultado):
        raise ValueError("El resultado no es finito.")
    return {
        "result": resultado,
        "operation": "dividir",
        "a": a,
        "b": b,
    }


@typechecked
def calcular_modulo(a: float, b: float) -> dict:
    if b == 0:
        raise ValueError("No se puede calcular módulo entre cero.")
    resultado = a % b
    if not math.isfinite(resultado):
        raise ValueError("El resultado no es finito.")
    return {
        "result": resultado,
        "operation": "modulo",
        "a": a,
        "b": b,
    }


@typechecked
def calcular_potencia(base: float, exponent: float) -> dict:
    resultado = base ** exponent

    if isinstance(resultado, complex):
        raise ValueError("No se admiten resultados complejos.")

    resultado = float(resultado)

    if not math.isfinite(resultado):
        raise ValueError("El resultado no es finito.")

    return {
        "result": resultado,
        "operation": "potencia",
        "base": base,
        "exponent": exponent,
    }


@typechecked
def calcular_raiz(n: float, x: float) -> dict:
    if n == 0:
        raise ValueError("El índice de la raíz no puede ser cero.")

    if x < 0:
        if n.is_integer() and int(n) % 2 != 0:
            resultado = -((-x) ** (1 / n))
        else:
            raise ValueError(
                "La raíz no tiene resultado real para estos valores."
            )
    else:
        resultado = x ** (1 / n)

    resultado = float(resultado)

    if not math.isfinite(resultado):
        raise ValueError("El resultado no es finito.")

    return {
        "result": resultado,
        "operation": "raiz",
        "n": n,
        "x": x,
    }


# =========================================================
# EVALUADOR SEGURO
# =========================================================

_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
}

_UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def evaluar_nodo(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return evaluar_nodo(node.body)

    if (
        isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and isinstance(node.value, (int, float))
    ):
        resultado = float(node.value)
        if not math.isfinite(resultado):
            raise ValueError("Número no finito.")
        return resultado

    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        a = evaluar_nodo(node.left)
        b = evaluar_nodo(node.right)

        if isinstance(node.op, (ast.Div, ast.Mod)) and b == 0:
            raise ValueError("No se permite dividir entre cero.")

        resultado = _BINOPS[type(node.op)](a, b)

        if isinstance(resultado, complex):
            raise ValueError("No se admiten resultados complejos.")

        resultado = float(resultado)

        if not math.isfinite(resultado):
            raise ValueError("El resultado no es finito.")

        return resultado

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        resultado = float(
            _UNARYOPS[type(node.op)](evaluar_nodo(node.operand))
        )
        if not math.isfinite(resultado):
            raise ValueError("El resultado no es finito.")
        return resultado

    raise ValueError("Expresión no permitida.")


@typechecked
def calcular_expresion(expr: str) -> dict:
    expr = expr.strip()

    if not expr:
        raise ValueError("La expresión está vacía.")

    try:
        resultado = evaluar_nodo(ast.parse(expr, mode="eval"))
        return {
            "result": resultado,
            "operation": "evaluar",
            "expr": expr,
        }
    except (
        SyntaxError,
        TypeError,
        OverflowError,
        ZeroDivisionError,
    ) as exc:
        raise ValueError(f"Expresión inválida: {exc}") from exc


# =========================================================
# FASTMCP: HERRAMIENTAS NATIVAS
# =========================================================

mcp = FastMCP("Calculadora")


def anotaciones(titulo: str) -> ToolAnnotations:
    return ToolAnnotations(
        title=titulo,
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )


@mcp.tool(
    name="sumar",
    description="Suma dos números.",
    annotations=anotaciones("Sumar"),
    meta={"service": "calculadora", "version": VERSION, "category": "matematicas", "tool": "sumar"}
)
@typechecked
@registrar_llamada("MCP", "sumar")
def tool_sumar(a: float, b: float) -> dict:
    return calcular_suma(a, b)


@mcp.tool(
    name="restar",
    description="Resta b de a.",
    annotations=anotaciones("Restar"),
    meta={"service": "calculadora", "version": VERSION, "category": "matematicas", "tool": "restar"}
)
@typechecked
@registrar_llamada("MCP", "restar")
def tool_restar(a: float, b: float) -> dict:
    return calcular_resta(a, b)


@mcp.tool(
    name="multiplicar",
    description="Multiplica dos números.",
    annotations=anotaciones("Multiplicar"),
    meta={"service": "calculadora", "version": VERSION, "category": "matematicas", "tool": "multiplicar"}
)
@typechecked
@registrar_llamada("MCP", "multiplicar")
def tool_multiplicar(a: float, b: float) -> dict:
    return calcular_multiplicacion(a, b)


@mcp.tool(
    name="dividir",
    description="Divide a entre b.",
    annotations=anotaciones("Dividir"),
    meta={"service": "calculadora", "version": VERSION, "category": "matematicas", "tool": "dividir"}
)
@typechecked
@registrar_llamada("MCP", "dividir")
def tool_dividir(a: float, b: float) -> dict:
    return calcular_division(a, b)


@mcp.tool(
    name="modulo",
    description="Calcula el resto de a dividido entre b.",
    annotations=anotaciones("Modulo"),
    meta={"service": "calculadora", "version": VERSION, "category": "matematicas", "tool": "modulo"}
)
@typechecked
@registrar_llamada("MCP", "modulo")
def tool_modulo(a: float, b: float) -> dict:
    return calcular_modulo(a, b)


@mcp.tool(
    name="potencia",
    description="Eleva una base a un exponente.",
    annotations=anotaciones("Potencia"),
    meta={"service": "calculadora", "version": VERSION, "category": "matematicas", "tool": "potencia"}
)
@typechecked
@registrar_llamada("MCP", "potencia")
def tool_potencia(base: float, exponent: float) -> dict:
    return calcular_potencia(base, exponent)


@mcp.tool(
    name="raiz",
    description="Calcula la raíz n-ésima de un número.",
    annotations=anotaciones("Raiz"),
    meta={"service": "calculadora", "version": VERSION, "category": "matematicas", "tool": "raiz"}
)
@typechecked
@registrar_llamada("MCP", "raiz")
def tool_raiz(n: float, x: float) -> dict:
    return calcular_raiz(n, x)


@mcp.tool(
    name="evaluar",
    description="Evalúa una expresión matemática de forma segura.",
    annotations=anotaciones("Evaluar"),
    meta={"service": "calculadora", "version": VERSION, "category": "matematicas", "tool": "evaluar"}
)
@typechecked
@registrar_llamada("MCP", "evaluar")
def tool_evaluar(expr: str) -> dict:
    return calcular_expresion(expr)


# =========================================================
# FASTAPI + CICLO DE VIDA MCP
# =========================================================

mcp_app = mcp.http_app(path="/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp_app.lifespan(app):
        yield


app = FastAPI(
    title="Math API WEB MCP",
    description="Microservicio matemático con API REST y MCP",
    version=VERSION,
    lifespan=lifespan,
)


# =========================================================
# ENDPOINTS REST
# =========================================================

def ejecutar_rest(nombre: str, funcion, *args):
    try:
        return funcion(*args)
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/sumar", tags=["Operaciones Básicas"], operation_id="rest_sumar")
@registrar_llamada("REST", "sumar")
def rest_sumar(operation: BinaryOperation):
    return ejecutar_rest("sumar", calcular_suma, operation.a, operation.b)


@app.post("/restar", tags=["Operaciones Básicas"], operation_id="rest_restar")
@registrar_llamada("REST", "restar")
def rest_restar(operation: BinaryOperation):
    return ejecutar_rest("restar", calcular_resta, operation.a, operation.b)


@app.post(
    "/multiplicar",
    tags=["Operaciones Básicas"],
    operation_id="rest_multiplicar",
)
@registrar_llamada("REST", "multiplicar")
def rest_multiplicar(operation: BinaryOperation):
    return ejecutar_rest(
        "multiplicar", calcular_multiplicacion, operation.a, operation.b
    )


@app.post("/dividir", tags=["Operaciones Básicas"], operation_id="rest_dividir")
@registrar_llamada("REST", "dividir")
def rest_dividir(operation: BinaryOperation):
    return ejecutar_rest("dividir", calcular_division, operation.a, operation.b)


@app.post("/modulo", tags=["Operaciones Básicas"], operation_id="rest_modulo")
@registrar_llamada("REST", "modulo")
def rest_modulo(operation: BinaryOperation):
    return ejecutar_rest("modulo", calcular_modulo, operation.a, operation.b)


@app.post(
    "/potencia",
    tags=["Operaciones Avanzadas"],
    operation_id="rest_potencia",
)
@registrar_llamada("REST", "potencia")
def rest_potencia(operation: PowerOperation):
    return ejecutar_rest(
        "potencia", calcular_potencia, operation.base, operation.exponent
    )


@app.post("/raiz", tags=["Operaciones Avanzadas"], operation_id="rest_raiz")
@registrar_llamada("REST", "raiz")
def rest_raiz(operation: RootOperation):
    return ejecutar_rest("raiz", calcular_raiz, operation.n, operation.x)


@app.post(
    "/evaluar",
    tags=["Operaciones Avanzadas"],
    operation_id="rest_evaluar",
)
@registrar_llamada("REST", "evaluar")
def rest_evaluar(operation: ExpressionOperation):
    return ejecutar_rest("evaluar", calcular_expresion, operation.expr)


# =========================================================
# INFORMACIÓN
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
# MONTAJE MCP
# =========================================================

app.mount("/mcp", mcp_app)


# =========================================================
# ARRANQUE
# =========================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
