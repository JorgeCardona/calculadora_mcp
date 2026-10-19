# Math API WEB MCP 🧮

Microservicio matemático híbrido construido con **FastAPI** y **FastMCP** (Model Context Protocol). Proporciona tanto una **API REST** para integración web estándar como un servidor **MCP** con transporte HTTP/SSE para asistentes de IA y consumo programático desde entornos como JupyterLab.

---

## 🚀 Características Principales

- **Arquitectura Híbrida**: Sirve endpoints REST (`/docs`) y un servidor MCP (`/mcp`) en el mismo puerto y aplicación FastAPI.
- **Validación Estricta y Seguridad**:
  - Modelos **Pydantic** con `strict=True` para prevenir coerciones indeseadas de datos.
  - Validación dinámica de tipos en tiempo de ejecución con `@typechecked` (`typeguard`).
  - **Evaluador Seguro de Expresiones Matemáticas**: Utiliza el módulo `ast` de Python para evaluar expresiones sin usar `eval()`.
- **Herramientas Disponibles**:
  - Operaciones básicas: `sumar`, `restar`, `multiplicar`, `dividir`, `modulo`.
  - Operaciones avanzadas: `potencia`, `raiz`, `evaluar` (expresiones matemáticas).

---

## 📡 Endpoints del Servidor

- **Documentación Swagger / REST**: https://calculadora-mcp.onrender.com/docs
- **Servidor MCP**: https://calculadora-mcp.onrender.com/mcp

---

## 🛠️ Herramientas MCP / Endpoints REST

| Herramienta | Tipo | Descripción | Parámetros |
| :--- | :--- | :--- | :--- |
| `sumar` | Básica | Suma dos números reales | `a: float`, `b: float` |
| `restar` | Básica | Resta `b` de `a` | `a: float`, `b: float` |
| `multiplicar` | Básica | Multiplica dos números | `a: float`, `b: float` |
| `dividir` | Básica | Divide `a` entre `b` (valida división por cero) | `a: float`, `b: float` |
| `modulo` | Básica | Calcula el resto de `a / b` | `a: float`, `b: float` |
| `potencia` | Avanzada | Eleva una base a un exponente | `base: float`, `exponent: float` |
| `raiz` | Avanzada | Raíz $n$-ésima de $x$ | `n: float`, `x: float` |
| `evaluar` | Avanzada | Evalúa expresiones compuestas de forma segura | `expr: str` |

---

## 💻 Consumo desde JupyterLab (Python)

### 1. LISTAR TOOLS
```
from mcp.client.streamable_http import streamable_http_client
from mcp.client.session import ClientSession

async def test_mcp():
    async with streamable_http_client(
        "https://calculadora-mcp.onrender.com/mcp"
    ) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            response = await session.list_tools()
            
            for tool in response.tools:
                print(f"Nombre: {tool.name}")
                print(f"Descripción: {tool.description}")
                print(f"Argumentos: {tool.input_schema}")
                print("-" * 50)

await test_mcp()
```

---

### 2. Consumo del Servidor MCP (`mcp` + `nest_asyncio`)

```python
import json
from mcp.client.streamable_http import streamable_http_client
from mcp.client.session import ClientSession

async with streamable_http_client(
    "https://calculadora-mcp.onrender.com/mcp"
) as (read, write):

    async with ClientSession(read, write) as session:
        await session.initialize()

        tools = await session.list_tools()
        print("Herramientas disponibles:")
        print([tool.name for tool in tools.tools])

        result = await session.call_tool(
            "sumar",
            arguments={"a": 5, "b": 10},
        )

        if result.is_error:
            print("Error MCP:", result.content)
        else:
            print("Respuesta:", result.content)
            for item in result.content:
                if item.type == "text":
                    data = json.loads(item.text)
                    print("Resultado:", data)

```

### 3. Consumo de la API REST (`requests`)

```python
import requests

BASE_URL = "https://calculadora-mcp.onrender.com"

# Probar endpoint REST /sumar
response = requests.post(
    f"{BASE_URL}/sumar",
    json={"a": 5.0, "b": 10.0}
)

print("Status Code:", response.status_code)
print("Respuesta REST:", response.json())

```
