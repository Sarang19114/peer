import React from 'react';
import MuiGrid from '@mui/material/GridLegacy';

type LegacyGridProps = React.ComponentProps<typeof MuiGrid>;

const LegacyGrid: React.FC<LegacyGridProps> = (props) => <MuiGrid {...props} />;

export default LegacyGrid;

