import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  CardActions,
  Chip,
  Button,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Stack,
} from '@mui/material';
import {
  Person as PersonIcon,
  LocationOn as LocationIcon,
  ExpandMore as ExpandMoreIcon,
  CheckCircle as CheckCircleIcon,
  TrendingUp as TrendingUpIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import type { MatchResult } from '../api/client';
import Grid from './common/LegacyGrid';

interface MatchResultsProps {
  matches: MatchResult[];
}

const MatchResults: React.FC<MatchResultsProps> = ({ matches }) => {
  const [expandedCard, setExpandedCard] = useState<string | false>(false);

  const handleExpand = (personId: string) => {
    setExpandedCard(expandedCard === personId ? false : personId);
  };

  const getMatchColor = (score: number) => {
    if (score >= 0.90) return 'success';
    if (score >= 0.85) return 'success';
    if (score >= 0.75) return 'info';
    if (score >= 0.65) return 'info';
    return 'warning';
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <Paper elevation={3} sx={{ p: 4, borderRadius: 3 }}>
      <Box mb={4}>
        <Typography variant="h4" component="h2" gutterBottom fontWeight="bold" align="center">
          Your Top Matches
        </Typography>
        <Typography variant="body1" color="text.secondary" align="center">
          We found {matches.length} compatible matches for you
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {matches.map((match, index) => (
          <Grid item xs={12} key={match.person_id}>
            <Card
              sx={{
                transition: 'transform 0.2s, box-shadow 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                  boxShadow: 6,
                },
                border: index === 0 ? '2px solid' : 'none',
                borderColor: index === 0 ? 'primary.main' : 'transparent',
              }}
            >
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="start" mb={2}>
                  <Box>
                    <Box display="flex" alignItems="center" gap={1} mb={1}>
                      <Typography variant="h5" component="h3" fontWeight="bold">
                        {match.name}
                      </Typography>
                      {index === 0 && (
                        <Chip
                          label="Best Match"
                          color="primary"
                          size="small"
                          sx={{ fontWeight: 'bold' }}
                        />
                      )}
                    </Box>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Match #{index + 1} • {match.person_id}
                    </Typography>
                  </Box>
                  <Chip
                    label={formatPercentage(match.match_probability)}
                    color={getMatchColor(match.match_probability)}
                    size="medium"
                    sx={{ fontWeight: 'bold', fontSize: '1rem' }}
                  />
                </Box>

                <Box mb={3}>
                  <LinearProgress
                    variant="determinate"
                    value={match.match_probability * 100}
                    color={getMatchColor(match.match_probability)}
                    sx={{ height: 10, borderRadius: 5 }}
                  />
                </Box>

                {match.compatibility_breakdown && (
                  <Grid container spacing={2} sx={{ mb: 3 }}>
                    {[
                      { label: 'Base Similarity', value: match.compatibility_breakdown.base_compatibility },
                      { label: 'Preference Fit', value: match.compatibility_breakdown.preference_multiplier },
                      { label: 'Intelligence Layers', value: match.compatibility_breakdown.layer_average },
                      { label: 'Final Score', value: match.compatibility_breakdown.final_score },
                    ].map((item) => (
                      <Grid item xs={6} sm={3} key={item.label}>
                        <Paper variant="outlined" sx={{ p: 1.5, textAlign: 'center', borderRadius: 2 }}>
                          <Typography variant="caption" color="text.secondary">
                            {item.label}
                          </Typography>
                          <Typography variant="subtitle1" fontWeight="bold">
                            {(item.value * 100).toFixed(0)}%
                          </Typography>
                        </Paper>
                      </Grid>
                    ))}
                  </Grid>
                )}

                {/* Explanation Summary */}
                <Box mb={3} sx={{ bgcolor: 'background.default', p: 2, borderRadius: 2 }}>
                  <Typography variant="subtitle2" fontWeight="bold" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <InfoIcon fontSize="small" />
                    Why This Match Works
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {match.explanation.summary}
                  </Typography>
                </Box>

                {/* Top Reasons */}
                {match.explanation.top_reasons && match.explanation.top_reasons.length > 0 && (
                  <Box mb={3}>
                    <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                      Key Compatibility Factors
                    </Typography>
                    <List dense>
                      {match.explanation.top_reasons.map((reason, idx) => (
                        <ListItem key={idx} disableGutters>
                          <ListItemIcon>
                            <CheckCircleIcon color="success" fontSize="small" />
                          </ListItemIcon>
                          <ListItemText primary={reason} />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                )}

                {/* Profile Summary */}
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6} sm={3}>
                    <Box display="flex" alignItems="center" gap={0.5}>
                      <PersonIcon fontSize="small" color="action" />
                      <Typography variant="body2">
                        Age: {match.age}
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Box display="flex" alignItems="center" gap={0.5}>
                      <LocationIcon fontSize="small" color="action" />
                      <Typography variant="body2" noWrap>
                        {match.location}
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="body2" color="text.secondary">
                      Gender: {match.gender}
                    </Typography>
                  </Grid>
                  <Grid item xs={6} sm={3}>
                    <Typography variant="body2" color="text.secondary">
                      Interest: {match.profile_summary.primary_interest || 'N/A'}
                    </Typography>
                  </Grid>
                  {match.profile_summary.communication_style && (
                    <Grid item xs={6} sm={3}>
                      <Typography variant="body2" color="text.secondary">
                        Communication: {match.profile_summary.communication_style}
                      </Typography>
                    </Grid>
                  )}
                  {match.profile_summary.years_experience && match.profile_summary.years_experience > 0 && (
                    <Grid item xs={6} sm={3}>
                      <Typography variant="body2" color="text.secondary">
                        Experience: {match.profile_summary.years_experience} years
                      </Typography>
                    </Grid>
                  )}
                </Grid>

                {/* Intelligence Layers */}
                {match.intelligence_layers && match.intelligence_layers.length > 0 && (
                  <Box mb={3}>
                    <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                      Intelligence Stack
                    </Typography>
                    <Grid container spacing={2}>
                      {match.intelligence_layers.map((layer) => (
                        <Grid item xs={12} md={6} key={layer.name}>
                          <Paper variant="outlined" sx={{ p: 2, height: '100%', borderRadius: 2 }}>
                            <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                              <Typography variant="body1" fontWeight="bold">
                                {layer.name}
                              </Typography>
                              <Chip
                                label={`${(layer.score * 100).toFixed(0)}%`}
                                color="primary"
                                size="small"
                                sx={{ fontWeight: 'bold' }}
                              />
                            </Box>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                              {layer.rationale}
                            </Typography>
                            {layer.signals?.length > 0 && (
                              <Stack spacing={0.5} sx={{ mb: layer.metadata?.predictions ? 1 : 0 }}>
                                {layer.signals.map((signal, idx) => (
                                  <Box
                                    key={`${layer.name}-${idx}`}
                                    display="flex"
                                    justifyContent="space-between"
                                    sx={{ fontSize: '0.85rem', color: 'text.secondary' }}
                                  >
                                    <span>{signal.label}</span>
                                    <strong>{signal.value}</strong>
                                  </Box>
                                ))}
                              </Stack>
                            )}
                            {Array.isArray(layer.metadata?.predictions) && layer.metadata.predictions.length > 0 && (
                              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                                {layer.metadata.predictions.map((prediction: { horizon: string; probability: number }) => (
                                  <Chip
                                    key={`${layer.name}-${prediction.horizon}`}
                                    label={`${prediction.horizon}: ${(prediction.probability * 100).toFixed(0)}%`}
                                    size="small"
                                    variant="outlined"
                                  />
                                ))}
                              </Stack>
                            )}
                          </Paper>
                        </Grid>
                      ))}
                    </Grid>
                  </Box>
                )}

                {/* Detailed Breakdown Accordion */}
                <Accordion expanded={expandedCard === match.person_id} onChange={() => handleExpand(match.person_id)}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="caption" fontWeight="bold">
                      View Detailed Compatibility Breakdown
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <TableContainer>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Compatibility Factor</TableCell>
                            <TableCell align="right">Score</TableCell>
                            <TableCell align="right">Level</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {match.explanation.detailed_breakdown.map((item, idx) => (
                            <TableRow key={idx}>
                              <TableCell>
                                <Typography variant="body2">
                                  {item.feature}
                                </Typography>
                              </TableCell>
                              <TableCell align="right">
                                <Box display="flex" alignItems="center" justifyContent="flex-end" gap={1}>
                                  <LinearProgress
                                    variant="determinate"
                                    value={item.score * 100}
                                    sx={{ width: 80, height: 6, borderRadius: 3 }}
                                    color={item.color as any}
                                  />
                                  <Typography variant="body2" sx={{ minWidth: 50 }}>
                                    {item.percentage}
                                  </Typography>
                                </Box>
                              </TableCell>
                              <TableCell align="right">
                                <Chip
                                  label={item.level}
                                  size="small"
                                  color={item.color as any}
                                  variant="outlined"
                                />
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </AccordionDetails>
                </Accordion>
              </CardContent>

              <CardActions sx={{ p: 2, pt: 0 }}>
                <Button size="small" startIcon={<TrendingUpIcon />}>
                  View Full Profile
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      {matches.length === 0 && (
        <Box textAlign="center" py={8}>
          <Typography variant="h6" color="text.secondary">
            No matches found. Please try adjusting your profile.
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default MatchResults;
