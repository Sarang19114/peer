import React, { useState } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stepper,
  Step,
  StepLabel,
  Chip,
  CircularProgress,
  Alert,
  Card,
  CardContent,
  Stack,
} from '@mui/material';
import {
  Person as PersonIcon,
  LocationOn as LocationIcon,
  FitnessCenter as FitnessIcon,
  Psychology as PsychologyIcon,
  Favorite as FavoriteIcon,
  Send as SendIcon,
  ArrowBack as ArrowBackIcon,
  ArrowForward as ArrowForwardIcon,
} from '@mui/icons-material';
import { apiClient } from '../api/client';
import type { UserProfile, UserPreferences } from '../api/client';
import MatchResults from './MatchResults';
import Grid from './common/LegacyGrid';

const steps = ['Your Profile', 'Your Preferences', 'Review & Match'];

interface EnhancedProfileFormProps {
  onMatchesFound?: (matches: any[]) => void;
}

type SectionCardProps = {
  icon: React.ElementType;
  title: string;
  description?: string;
  children: React.ReactNode;
};

const SectionCard: React.FC<SectionCardProps> = ({ icon: Icon, title, description, children }) => (
  <Card variant="outlined" sx={{ height: '100%', borderRadius: 3 }}>
    <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Box display="flex" alignItems="center" gap={1}>
        <Icon color="primary" />
        <Box>
          <Typography variant="subtitle1" fontWeight="bold">
            {title}
          </Typography>
          {description && (
            <Typography variant="body2" color="text.secondary">
              {description}
            </Typography>
          )}
        </Box>
      </Box>
      {children}
    </CardContent>
  </Card>
);

const EnhancedProfileForm: React.FC<EnhancedProfileFormProps> = ({ onMatchesFound }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [matches, setMatches] = useState<any[]>([]);

  // Profile state
  const [age, setAge] = useState<number>(30);
  const [location, setLocation] = useState('');
  const [gender, setGender] = useState('');
  const [totalDistanceKm, setTotalDistanceKm] = useState<number>(0);
  const [totalActivities, setTotalActivities] = useState<number>(0);
  const [recoveryScore, setRecoveryScore] = useState<number>(70);
  const [socialEngagement, setSocialEngagement] = useState<number>(7);
  const [communicationPreference, setCommunicationPreference] = useState('');
  const [primaryInterest, setPrimaryInterest] = useState('');
  const [experienceLevel, setExperienceLevel] = useState('');
  const [injuryCount, setInjuryCount] = useState<number>(0);
  const [yearsExperience, setYearsExperience] = useState<number>(0);

  // Preferences state
  const [preferredAgeMin, setPreferredAgeMin] = useState<number>(25);
  const [preferredAgeMax, setPreferredAgeMax] = useState<number>(45);
  const [preferredGender, setPreferredGender] = useState('');
  const [preferredLocation, setPreferredLocation] = useState('');
  const [preferredInterests, setPreferredInterests] = useState<string[]>([]);
  const [preferredExperienceLevel, setPreferredExperienceLevel] = useState('');
  const [preferredCommunicationStyle, setPreferredCommunicationStyle] = useState('');

  const handleNext = () => {
    if (activeStep === 0) {
      if (!age || !location || !gender) {
        setError('Please fill in all required fields (age, location, gender)');
        return;
      }
    }
    setError(null);
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setError(null);
    setActiveStep((prevStep) => prevStep - 1);
  };

  const handleSubmit = async () => {
    try {
      setLoading(true);
      setError(null);

      const profile: UserProfile = {
        age,
        location,
        gender,
        total_distance_km: totalDistanceKm || undefined,
        total_activities: totalActivities || undefined,
        avg_daily_recovery_score: recoveryScore || undefined,
        social_engagement_score: socialEngagement || undefined,
        communication_preference: communicationPreference || undefined,
        primary_interest: primaryInterest || undefined,
        experience_level: experienceLevel || undefined,
        injury_count: injuryCount || undefined,
        years_experience: yearsExperience || undefined,
      };

      const preferences: UserPreferences = {
        preferred_age_min: preferredAgeMin || undefined,
        preferred_age_max: preferredAgeMax || undefined,
        preferred_gender: preferredGender || undefined,
        preferred_location: preferredLocation || undefined,
        preferred_interests: preferredInterests.length > 0 ? preferredInterests : undefined,
        preferred_experience_level: preferredExperienceLevel || undefined,
        preferred_communication_style: preferredCommunicationStyle || undefined,
      };

      const response = await apiClient.findMatches(profile, preferences);
      setMatches(response.matches);
      if (onMatchesFound) {
        onMatchesFound(response.matches);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to find matches');
      setMatches([]);
    } finally {
      setLoading(false);
    }
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <Stack spacing={3}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={7}>
                <SectionCard
                  icon={PersonIcon}
                  title="Identity & Locale"
                  description="Set the fundamentals so we can anchor you in the right network."
                >
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        required
                        label="Age"
                        type="number"
                        value={age}
                        onChange={(e) => setAge(parseInt(e.target.value, 10) || 18)}
                        inputProps={{ min: 18, max: 100 }}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        required
                        label="Location"
                        value={location}
                        onChange={(e) => setLocation(e.target.value)}
                        placeholder="City, Country"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <FormControl fullWidth required>
                        <InputLabel>Gender</InputLabel>
                        <Select
                          value={gender}
                          onChange={(e) => setGender(e.target.value)}
                          label="Gender"
                        >
                          <MenuItem value="Male">Male</MenuItem>
                          <MenuItem value="Female">Female</MenuItem>
                          <MenuItem value="Non-binary">Non-binary</MenuItem>
                          <MenuItem value="Prefer not to say">Prefer not to say</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                  </Grid>
                </SectionCard>
              </Grid>
              <Grid item xs={12} md={5}>
                <SectionCard
                  icon={FitnessIcon}
                  title="Performance Snapshot"
                  description="Quantify your current training rhythm."
                >
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Distance (km)"
                        type="number"
                        value={totalDistanceKm || ''}
                        onChange={(e) => setTotalDistanceKm(parseFloat(e.target.value) || 0)}
                        helperText="Lifetime or YTD volume"
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Total Activities"
                        type="number"
                        value={totalActivities || ''}
                        onChange={(e) => setTotalActivities(parseInt(e.target.value, 10) || 0)}
                        helperText="Sessions logged"
                      />
                    </Grid>
                  </Grid>
                  <Box>
                    <Typography gutterBottom>Recovery Score</Typography>
                    <Slider
                      value={recoveryScore}
                      onChange={(_, value) => setRecoveryScore(value as number)}
                      min={0}
                      max={100}
                      step={1}
                      valueLabelDisplay="auto"
                    />
                  </Box>
                </SectionCard>
              </Grid>
              <Grid item xs={12} md={6}>
                <SectionCard
                  icon={PsychologyIcon}
                  title="Focus & Experience"
                  description="Show us where you thrive so we can find peers on the same trajectory."
                >
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Primary Interest"
                        value={primaryInterest}
                        onChange={(e) => setPrimaryInterest(e.target.value)}
                        placeholder="e.g., Running, Cycling"
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth>
                        <InputLabel>Experience Level</InputLabel>
                        <Select
                          value={experienceLevel}
                          onChange={(e) => setExperienceLevel(e.target.value)}
                          label="Experience Level"
                        >
                          <MenuItem value="">Not specified</MenuItem>
                          <MenuItem value="Beginner">Beginner</MenuItem>
                          <MenuItem value="Intermediate">Intermediate</MenuItem>
                          <MenuItem value="Advanced">Advanced</MenuItem>
                          <MenuItem value="Elite Amateur">Elite Amateur</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Years of Experience"
                        type="number"
                        value={yearsExperience || ''}
                        onChange={(e) => setYearsExperience(parseInt(e.target.value, 10) || 0)}
                      />
                    </Grid>
                  </Grid>
                </SectionCard>
              </Grid>
              <Grid item xs={12} md={6}>
                <SectionCard
                  icon={FavoriteIcon}
                  title="Communication & Recovery"
                  description="Match faster by aligning interaction styles and health context."
                >
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth>
                        <InputLabel>Communication Preference</InputLabel>
                        <Select
                          value={communicationPreference}
                          onChange={(e) => setCommunicationPreference(e.target.value)}
                          label="Communication Preference"
                        >
                          <MenuItem value="">Not specified</MenuItem>
                          <MenuItem value="Text-focused">Text-focused</MenuItem>
                          <MenuItem value="Video Sessions">Video Sessions</MenuItem>
                          <MenuItem value="Group Activities">Group Activities</MenuItem>
                          <MenuItem value="In-person">In-person</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Past Injuries"
                        type="number"
                        value={injuryCount || ''}
                        onChange={(e) => setInjuryCount(parseInt(e.target.value, 10) || 0)}
                        inputProps={{ min: 0 }}
                      />
                    </Grid>
                  </Grid>
                  <Box>
                    <Typography gutterBottom>Social Engagement (0-10)</Typography>
                    <Slider
                      value={socialEngagement}
                      onChange={(_, value) => setSocialEngagement(value as number)}
                      min={0}
                      max={10}
                      step={0.1}
                      marks={[
                        { value: 0, label: 'Low' },
                        { value: 5, label: 'Medium' },
                        { value: 10, label: 'High' },
                      ]}
                      valueLabelDisplay="auto"
                    />
                  </Box>
                </SectionCard>
              </Grid>
            </Grid>

            <Box display="flex" flexWrap="wrap" gap={1}>
              <Chip label={`Age ${age}`} variant="outlined" color="primary" />
              {primaryInterest && <Chip label={`Focus: ${primaryInterest}`} variant="outlined" />}
              {experienceLevel && <Chip label={`Level: ${experienceLevel}`} variant="outlined" />}
              {location && <Chip label={location} variant="outlined" />}
              {communicationPreference && <Chip label={`Prefers ${communicationPreference}`} variant="outlined" />}
            </Box>
          </Stack>
        );

      case 1:
        return (
          <Stack spacing={3}>
            <Typography variant="body1" color="text.secondary">
              Tell us what “great” looks like. Every filter reshapes the compatibility engine.
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <SectionCard
                  icon={FavoriteIcon}
                  title="Match Guardrails"
                  description="Define the age and gender window where collaboration works best."
                >
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Minimum Age"
                        type="number"
                        value={preferredAgeMin || ''}
                        onChange={(e) => setPreferredAgeMin(parseInt(e.target.value, 10) || 18)}
                        inputProps={{ min: 18, max: 100 }}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        fullWidth
                        label="Maximum Age"
                        type="number"
                        value={preferredAgeMax || ''}
                        onChange={(e) => setPreferredAgeMax(parseInt(e.target.value, 10) || 100)}
                        inputProps={{ min: 18, max: 100 }}
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <FormControl fullWidth>
                        <InputLabel>Preferred Gender</InputLabel>
                        <Select
                          value={preferredGender}
                          onChange={(e) => setPreferredGender(e.target.value)}
                          label="Preferred Gender"
                        >
                          <MenuItem value="">Any</MenuItem>
                          <MenuItem value="Male">Male</MenuItem>
                          <MenuItem value="Female">Female</MenuItem>
                          <MenuItem value="Non-binary">Non-binary</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                  </Grid>
                </SectionCard>
              </Grid>
              <Grid item xs={12} md={6}>
                <SectionCard
                  icon={LocationIcon}
                  title="Geography & Interests"
                  description="Blend local proximity with distributed network reach."
                >
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Preferred Location"
                        value={preferredLocation}
                        onChange={(e) => setPreferredLocation(e.target.value)}
                        placeholder="City, Country (optional)"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <TextField
                        fullWidth
                        label="Preferred Interests"
                        value={preferredInterests.join(', ')}
                        onChange={(e) => {
                          const interests = e.target.value
                            .split(',')
                            .map((i) => i.trim())
                            .filter((i) => i);
                          setPreferredInterests(interests);
                        }}
                        placeholder="e.g., Running, Cycling"
                        helperText="Comma-separated list"
                      />
                    </Grid>
                  </Grid>
                </SectionCard>
              </Grid>
              <Grid item xs={12} md={6}>
                <SectionCard
                  icon={PsychologyIcon}
                  title="Experience & Style"
                  description="Keep the collaboration cadence consistent."
                >
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth>
                        <InputLabel>Experience Level</InputLabel>
                        <Select
                          value={preferredExperienceLevel}
                          onChange={(e) => setPreferredExperienceLevel(e.target.value)}
                          label="Experience Level"
                        >
                          <MenuItem value="">Any</MenuItem>
                          <MenuItem value="Beginner">Beginner</MenuItem>
                          <MenuItem value="Intermediate">Intermediate</MenuItem>
                          <MenuItem value="Advanced">Advanced</MenuItem>
                          <MenuItem value="Elite Amateur">Elite Amateur</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth>
                        <InputLabel>Communication Style</InputLabel>
                        <Select
                          value={preferredCommunicationStyle}
                          onChange={(e) => setPreferredCommunicationStyle(e.target.value)}
                          label="Communication Style"
                        >
                          <MenuItem value="">Any</MenuItem>
                          <MenuItem value="Text-focused">Text-focused</MenuItem>
                          <MenuItem value="Video Sessions">Video Sessions</MenuItem>
                          <MenuItem value="Group Activities">Group Activities</MenuItem>
                          <MenuItem value="In-person">In-person</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                  </Grid>
                </SectionCard>
              </Grid>
            </Grid>
            <Box display="flex" flexWrap="wrap" gap={1}>
              <Chip
                label={`Age Range: ${preferredAgeMin || '18'}-${preferredAgeMax || '100'}`}
                variant="outlined"
                color="primary"
              />
              {preferredGender && <Chip label={`Gender: ${preferredGender}`} variant="outlined" />}
              {preferredLocation && <Chip label={`Location: ${preferredLocation}`} variant="outlined" />}
              {preferredInterests.length > 0 && <Chip label={`Interests: ${preferredInterests.join(', ')}`} variant="outlined" />}
              {preferredExperienceLevel && <Chip label={`Level: ${preferredExperienceLevel}`} variant="outlined" />}
            </Box>
          </Stack>
        );

      case 2:
        return (
          <Stack spacing={3}>
            <Typography variant="body1" color="text.secondary">
              Confirm the recap below. We’ll feed every detail into the transparent compatibility layers.
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <SectionCard
                  icon={PersonIcon}
                  title="Profile Recap"
                  description="Core signals driving peer discovery."
                >
                  <Stack spacing={1}>
                    <Typography variant="body2">Age: {age}</Typography>
                    <Typography variant="body2">Location: {location || 'Not provided'}</Typography>
                    <Typography variant="body2">Gender: {gender}</Typography>
                    <Typography variant="body2">Primary Interest: {primaryInterest || 'Not provided'}</Typography>
                    <Typography variant="body2">Experience Level: {experienceLevel || 'Not provided'}</Typography>
                    <Typography variant="body2">
                      Training Volume: {totalActivities || 0} sessions • {totalDistanceKm || 0} km
                    </Typography>
                    <Typography variant="body2">
                      Recovery & Social: {recoveryScore}% recovery • {socialEngagement}/10 social
                    </Typography>
                  </Stack>
                </SectionCard>
              </Grid>
              <Grid item xs={12} md={6}>
                <SectionCard
                  icon={FavoriteIcon}
                  title="Preference Recap"
                  description="Filters that reshape compatibility math."
                >
                  <Stack spacing={1}>
                    <Typography variant="body2">
                      Age Range: {preferredAgeMin || 18} - {preferredAgeMax || 100}
                    </Typography>
                    <Typography variant="body2">Gender: {preferredGender || 'Any'}</Typography>
                    <Typography variant="body2">Location: {preferredLocation || 'Flexible'}</Typography>
                    <Typography variant="body2">
                      Interests: {preferredInterests.length > 0 ? preferredInterests.join(', ') : 'Open'}
                    </Typography>
                    <Typography variant="body2">
                      Experience: {preferredExperienceLevel || 'Any'} • Communication:{' '}
                      {preferredCommunicationStyle || 'Any'}
                    </Typography>
                  </Stack>
                </SectionCard>
              </Grid>
            </Grid>
            <Paper variant="outlined" sx={{ p: 3, borderRadius: 3 }}>
              <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                What happens next
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                We’ll calculate distributed geographic reach, collaborative interest overlap, graph cohorts, and
                temporal success predictions. Expect only &gt;90% compatibility matches with full transparency.
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1}>
                <Chip label="Geographic Network Optimization" variant="outlined" />
                <Chip label="Temporal Success Prediction" variant="outlined" />
                <Chip label="Collaborative Filtering" variant="outlined" />
                <Chip label="Graph Cohort Discovery" variant="outlined" />
              </Box>
            </Paper>
          </Stack>
        );

      default:
        return null;
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: { xs: 3, md: 6 } }}>
      <Paper elevation={3} sx={{ p: { xs: 3, md: 5 }, borderRadius: 4 }}>
        <Stack spacing={2} alignItems="center" textAlign="center" mb={4}>
          <Chip label="Intelligence-driven matchmaking" color="primary" variant="outlined" />
          <Typography variant="h4" component="h1" fontWeight="bold">
            Find Your Perfect Match
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Clean inputs, transparent scoring. Every field powers geographic networks, collaborative filtering,
            temporal forecasts, and graph cohorts.
          </Typography>
        </Stack>

        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        <Box sx={{ minHeight: { xs: 'auto', md: '540px' }, mb: 4, width: '100%' }}>
          {renderStepContent(activeStep)}
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
          <Button
            disabled={activeStep === 0}
            onClick={handleBack}
            size="large"
            startIcon={<ArrowBackIcon />}
          >
            Back
          </Button>
          {activeStep === steps.length - 1 ? (
            <Button
              variant="contained"
              onClick={handleSubmit}
              disabled={loading}
              size="large"
              startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
            >
              {loading ? 'Finding Matches...' : 'Find My Matches'}
            </Button>
          ) : (
            <Button
              variant="contained"
              onClick={handleNext}
              size="large"
              endIcon={<ArrowForwardIcon />}
            >
              Next
            </Button>
          )}
        </Box>
      </Paper>

      {matches.length > 0 && (
        <Box sx={{ mt: 4 }}>
          <MatchResults matches={matches} />
        </Box>
      )}
    </Container>
  );
};

export default EnhancedProfileForm;

