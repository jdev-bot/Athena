"""DNA version persistence — immutable snapshots of strategy DNA vectors."""
import logging
from typing import Optional, List, Dict, Any

from athena.services.models import get_session, DnaSnapshot

logger = logging.getLogger(__name__)


def snapshot_dna(strategy_id: str, dna_vector: dict, source: str = "manual") -> int:
    """Persist a new DNA snapshot for a strategy. Returns the assigned version number."""
    session = get_session()
    latest = (
        session.query(DnaSnapshot)
        .filter_by(strategy_id=strategy_id)
        .order_by(DnaSnapshot.version.desc())
        .first()
    )
    version = (latest.version + 1) if latest else 1
    snap = DnaSnapshot(
        strategy_id=strategy_id,
        version=version,
        dna_vector=dict(dna_vector),
        source=source,
    )
    session.add(snap)
    session.commit()
    session.close()
    logger.info("DNA snapshot v%d saved for %s (source=%s)", version, strategy_id, source)
    return version


def restore_dna(strategy_id: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Restore a DNA vector snapshot. Returns None if not found."""
    session = get_session()
    if version is None:
        row = (
            session.query(DnaSnapshot)
            .filter_by(strategy_id=strategy_id)
            .order_by(DnaSnapshot.version.desc())
            .first()
        )
    else:
        row = session.query(DnaSnapshot).filter_by(strategy_id=strategy_id, version=version).first()
    session.close()
    if row is None:
        return None
    return dict(row.dna_vector)


def list_snapshots(strategy_id: str) -> List[Dict[str, Any]]:
    """List all DNA snapshots for a strategy."""
    session = get_session()
    rows = (
        session.query(DnaSnapshot)
        .filter_by(strategy_id=strategy_id)
        .order_by(DnaSnapshot.version.asc())
        .all()
    )
    session.close()
    return [
        {
            "version": r.version,
            "source": r.source,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "dna_preview": {
                k: v for i, (k, v) in enumerate(r.dna_vector.items()) if i < 5
            },
        }
        for r in rows
    ]
