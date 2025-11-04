"""Activity per grid cell over a period."""
from __future__ import annotations

import pandas as pd

from backend.app.database import db
from backend.app.services.areas import Selection
from ml import config


def cell_values(sel: Selection, start: str, end: str, layer: str) -> pd.DataFrame:
    """Count, FRP and HFAI of every cell of the selection between two dates.

    layer: "harmonized" (model estimate plus background rate), "modis" or
    "viirs" (observations). Observed layers only return cells with detections.
    """
    cur = db.cursor()
    if sel.kind == "aoi":
        cur.register("sel_cells", pd.DataFrame({"cell_id": sel.cell_ids}))
        cells_sql, params = "SELECT cell_id FROM sel_cells", []
    else:
        cells_sql, params = "SELECT cell_id FROM area_cells WHERE area_id = ?", [sel.area_id]

    if layer == "harmonized":
        days = pd.date_range(start, end, freq="D")
        month_days = pd.Series(days.month).value_counts().rename_axis("month").reset_index(name="n")
        cur.register("month_days", month_days)
        sql = f"""
            WITH sel AS ({cells_sql}),
            est AS (
                SELECT cell_id, sum(est_count - bg_count) AS c, sum(est_frp - bg_frp) AS f
                FROM harmonized_activity
                WHERE date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE) AND cell_id IN (SELECT cell_id FROM sel)
                GROUP BY cell_id
            ),
            bg AS (
                SELECT b.cell_id, sum(b.bg_count * m.n) AS c, sum(b.bg_frp * m.n) AS f
                FROM background_activity b JOIN month_days m USING (month)
                WHERE b.cell_id IN (SELECT cell_id FROM sel)
                GROUP BY b.cell_id
            )
            SELECT g.cell_id, g.ix, g.iy, g.lon, g.lat,
                   coalesce(est.c, 0) + coalesce(bg.c, 0) AS count,
                   coalesce(est.f, 0) + coalesce(bg.f, 0) AS frp
            FROM grid_cells g JOIN sel USING (cell_id)
            LEFT JOIN est USING (cell_id) LEFT JOIN bg USING (cell_id)
        """
    else:
        sql = f"""
            WITH sel AS ({cells_sql})
            SELECT g.cell_id, g.ix, g.iy, g.lon, g.lat,
                   sum(o.{layer}_count)::DOUBLE AS count, sum(o.{layer}_frp)::DOUBLE AS frp
            FROM observed_activity o JOIN grid_cells g USING (cell_id)
            WHERE o.date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE) AND o.cell_id IN (SELECT cell_id FROM sel)
            GROUP BY g.cell_id, g.ix, g.iy, g.lon, g.lat
            HAVING sum(o.{layer}_count) > 0
        """
    frame = cur.execute(sql, params + [start, end]).df()
    frame["hfai"] = config.hfai(frame["count"], frame["frp"], db.metadata()["hfai"])
    return frame
