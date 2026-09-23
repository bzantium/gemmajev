"""Lossless text layouts for the same visible game observation."""


def format_rows(rows, layout="original"):
    """Format every visible cell; never derive answers or expose hidden cells."""
    if layout == "original":
        return rows
    if layout != "indexed_grid":
        raise ValueError(f"Unknown observation layout: {layout}")
    formatted = []
    for row in rows:
        state = row["state"]
        marker = "Local map:\n"
        if marker not in state:
            formatted.append(row)
            continue
        prefix, text = state.split(marker, 1)
        grid = text.splitlines()
        if (len(grid) != 5 or any(len(line) != 5 or set(line) - set(".#AX") for line in grid)
                or grid[2][2] != "A"):
            raise ValueError("Expected the original centered 5 by 5 local observation")
        table = "     c0 c1 c2 c3 c4\n" + "\n".join(
            f"r{index}:  " + "  ".join(line) for index, line in enumerate(grid)
        )
        state = (prefix + "Local grid indices run from 0 to 4. "
                 "The agent is at local row 2, column 2.\n" + marker + table)
        formatted.append(dict(row, state=state))
    return formatted
