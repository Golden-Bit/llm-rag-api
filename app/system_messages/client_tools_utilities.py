from __future__ import annotations
import json
from enum import Enum
from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field


# ╭──────────────────────────────────────────────────────────────╮
# │ ENUM con i tipi permessi                                     │
# ╰──────────────────────────────────────────────────────────────╯
class ParamType(str, Enum):
    string  = "string"
    integer = "int"
    number  = "float"
    boolean = "bool"
    array   = "list"
    object  = "dict"


# ╭──────────────────────────────────────────────────────────────╮
# │ 1 · Parametri (ricorsivi)                                    │
# ╰──────────────────────────────────────────────────────────────╯
class ToolParamSpec(BaseModel):
    # attributi base
    name       : str       = Field(..., description="Nome del parametro JSON")
    param_type : ParamType = Field(..., description="Tipo primario")

    # tipi interni per list / dict
    items_type : Optional[ParamType] = Field(None, description="Tipo elementi se list")
    key_type   : Optional[ParamType] = Field(None, description="Tipo chiavi se dict")
    value_type : Optional[ParamType] = Field(None, description="Tipo valori se dict")

    description    : str           = Field(..., description="Spiegazione")
    example        : Optional[Any] = Field(None, description="Esempio lecito")
    default        : Optional[Any] = Field(None, description="Valore di default")
    allowed_values : Optional[List[Any]] = Field(None, description="Valori ammessi")

    # vincoli
    min_value  : Optional[float] = Field(None, description="Valore minimo")
    max_value  : Optional[float] = Field(None, description="Valore massimo")
    min_length : Optional[int]   = Field(None, description="Lunghezza minima")
    max_length : Optional[int]   = Field(None, description="Lunghezza massima")

    # annidamento
    properties : Optional[List["ToolParamSpec"]] = Field(
        None, description="Sotto-campi se param_type == dict"
    )

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def to_json(self) -> Dict[str, Any]:
        """
        Restituisce un dizionario JSON-serializzabile con tutti i campi di ToolSpec,
        inclusi i parametri (con le loro proprietà).
        """
        # pydantic.dict() già fa la conversione ricorsiva dei BaseModel
        return self.dict(
            by_alias    = True,
            exclude_none= True,
        )

# ╭──────────────────────────────────────────────────────────────╮
# │ 2 · ToolSpec                                                 │
# ╰──────────────────────────────────────────────────────────────╯
class ToolSpec(BaseModel):
    tool_name   : str  = Field(..., description="Nome del tool/widget")
    description : str  = Field(..., description="Descrizione generale")
    params      : List[ToolParamSpec] = Field(default_factory=list)

    # ────────────────────────────────────────────────────────────
    # helpers JSON (example / default)
    # ────────────────────────────────────────────────────────────
    def _build_json(self, mode: str) -> Dict[str, Any]:
        def _gen(p: ToolParamSpec):
            # sorgente valore
            if mode == "example" and p.example is not None:
                val = p.example
            elif mode == "default" and p.default is not None:
                val = p.default
            else:
                val = {
                    ParamType.string:  "",
                    ParamType.integer: 0,
                    ParamType.number:  0.0,
                    ParamType.boolean: False,
                    ParamType.array:   [],
                    ParamType.object:  {},
                }[p.param_type]
            # ricorsione
            if p.properties:
                val = {sub.name: _gen(sub) for sub in p.properties}
            return val
        return {p.name: _gen(p) for p in self.params}

    def example_json(self) -> Dict[str, Any]:
        return self._build_json("example")

    def default_json(self) -> Dict[str, Any]:
        return self._build_json("default")

    # ────────────────────────────────────────────────────────────
    # NEW 1 : descrizione schema (ricorsiva)
    # ────────────────────────────────────────────────────────────
    def schema_description(self) -> str:
        """Restituisce markdown descrittivo dello schema d’ingresso."""
        def _bounds(p: ToolParamSpec) -> str:
            b = []
            if p.min_value is not None:  b.append(f"min={p.min_value}")
            if p.max_value is not None:  b.append(f"max={p.max_value}")
            if p.min_length is not None: b.append(f"min_len={p.min_length}")
            if p.max_length is not None: b.append(f"max_len={p.max_length}")
            return f" ({', '.join(b)})" if b else ""

        def _render(p: ToolParamSpec, indent: int = 0) -> List[str]:
            tab = "  " * indent
            allowed = f" (valori ammessi: {p.allowed_values})" if p.allowed_values else ""
            default = f" [default: {p.default}]"              if p.default is not None else ""
            example = f" ‒ esempio: `{p.example}`"            if p.example is not None else ""
            rows = [f"{tab}* **{p.name}**: `{p.param_type.value}`"
                    f"{_bounds(p)}{allowed}{default}\n"
                    f"{tab}  – {p.description}{example}"]
            if p.properties:
                rows.append(f"{tab}  _Parametri interni_:")
                for sub in p.properties:
                    rows.extend(_render(sub, indent + 1))
            return rows

        lines: List[str] = ["#### Schema parametri"]
        for prm in self.params:
            lines.extend(_render(prm))
        return "\n".join(lines)

    # ────────────────────────────────────────────────────────────
    # NEW 2 : istruzioni complete (include schema)
    # ────────────────────────────────────────────────────────────
    def build_widget_instructions(
        self,
        *,
        open_char: str = "<",
        close_char: str = ">",
        sep_char: str = "|"
    ) -> str:
        json_block = json.dumps(self.example_json(), indent=2, ensure_ascii=False)
        start_tag  = f"{open_char} TYPE='WIDGET' WIDGET_ID='{self.tool_name}'"
        end_tag    = f"TYPE='WIDGET' WIDGET_ID='{self.tool_name}' {close_char}"

        return "\n".join([
            "---",
            f"**Istruzioni per lo strumento “{self.tool_name}”**",
            "",
            self.description,
            "",
            self.schema_description(),
            "",
            "#### Blocco da inserire (senza back-tick):",
            f"{start_tag} {sep_char} {json_block} {sep_char} {end_tag}",
            "",
            "Assicurati che:",
            f"- il blocco inizi con `{start_tag} {sep_char}` e termini con `{sep_char} {end_tag}`;",
            "- il JSON sia conforme allo schema sopra;",
            "- non ci siano caratteri estranei fuori dal blocco.",
            "---"
        ])

#from fastapi import FastAPI
#from typing import List

#app = FastAPI()

#@app.post("/configure_chain/")
#async def configure_chain(tool_specs: List[ToolSpec]):
#    """
#    Riceve una lista di ToolSpec e li usa per costruire il prompt,
#    configurare la chain, ecc.
#    """
#    # qui puoi accedere a tool_specs[0].tool_name, ecc.
#    return {"received": len(tool_specs)}

# ╭──────────────────────────────────────────────────────────────╮
# │ 3 · Self-test                                                │
# ╰──────────────────────────────────────────────────────────────╯
if __name__ == "__main__":
    loan_tool = ToolSpec(
        tool_name="LoanCalculator",
        description="Calcola rata, interessi e piano ammortamento di un mutuo.",
        params=[
            ToolParamSpec(
                name="amount", param_type=ParamType.number,
                description="Capitale richiesto (EUR).",
                example=250_000, min_value=10_000, max_value=2_000_000
            ),
            ToolParamSpec(
                name="duration_years", param_type=ParamType.integer,
                description="Durata del prestito in anni.",
                default=20, min_value=1, max_value=40
            ),
            ToolParamSpec(
                name="rate_options", param_type=ParamType.object,
                description="Opzioni avanzate per il tasso.",
                properties=[
                    ToolParamSpec(
                        name="type", param_type=ParamType.string,
                        description="Tipologia di tasso.",
                        allowed_values=["fixed", "variable"],
                        default="fixed"
                    ),
                    ToolParamSpec(
                        name="spread", param_type=ParamType.number,
                        description="Spread aggiuntivo (bps).",
                        example=1.5, min_value=0.1, max_value=5.0
                    ),
                ]
            )
        ]
    )

    print("=== JSON ESEMPIO ===")
    print(json.dumps(loan_tool.example_json(), indent=2, ensure_ascii=False))
    print("\n=== JSON DEFAULT ===")
    print(json.dumps(loan_tool.default_json(), indent=2, ensure_ascii=False))
    print("\n=== DESCRIZIONE SCHEMA ===")
    print(loan_tool.schema_description())
    print("\n=== ISTRUZIONI COMPLETE ===")
    print(loan_tool.build_widget_instructions())
