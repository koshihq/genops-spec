{{/*
Expand the name of the chart.
*/}}
{{- define "genops-ai.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "genops-ai.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "genops-ai.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "genops-ai.labels" -}}
helm.sh/chart: {{ include "genops-ai.chart" . }}
{{ include "genops-ai.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "genops-ai.selectorLabels" -}}
app.kubernetes.io/name: {{ include "genops-ai.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "genops-ai.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "genops-ai.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the config map to use
*/}}
{{- define "genops-ai.configMapName" -}}
{{- if .Values.configMap.create }}
{{- default (printf "%s-config" (include "genops-ai.fullname" .)) .Values.configMap.name }}
{{- else }}
{{- .Values.configMap.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the secret to use
*/}}
{{- define "genops-ai.secretName" -}}
{{- if .Values.secrets.create }}
{{- default (printf "%s-secrets" (include "genops-ai.fullname" .)) .Values.secrets.name }}
{{- else }}
{{- .Values.secrets.name }}
{{- end }}
{{- end }}

{{/*
Get image repository with global registry
*/}}
{{- define "genops-ai.image" -}}
{{- $registry := .Values.global.imageRegistry | default "" }}
{{- $repository := .Values.deployment.image.repository }}
{{- $tag := .Values.deployment.image.tag | default .Chart.AppVersion }}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}

{{/*
Environment-specific resource overrides
*/}}
{{- define "genops-ai.resources" -}}
{{- $environment := .Values.global.environment }}
{{- $envConfig := index .Values.environments $environment }}
{{- if and $envConfig $envConfig.resources }}
{{- toYaml $envConfig.resources }}
{{- else }}
{{- toYaml .Values.deployment.container.resources }}
{{- end }}
{{- end }}

{{/*
Environment-specific replica count
*/}}
{{- define "genops-ai.replicaCount" -}}
{{- $environment := .Values.global.environment }}
{{- $envConfig := index .Values.environments $environment }}
{{- if and $envConfig $envConfig.replicaCount }}
{{- $envConfig.replicaCount }}
{{- else }}
{{- .Values.deployment.replicaCount }}
{{- end }}
{{- end }}

{{/*
Environment-specific autoscaling configuration
*/}}
{{- define "genops-ai.autoscaling" -}}
{{- $environment := .Values.global.environment }}
{{- $envConfig := index .Values.environments $environment }}
{{- if and $envConfig $envConfig.autoscaling }}
{{- toYaml $envConfig.autoscaling }}
{{- else }}
{{- toYaml .Values.autoscaling }}
{{- end }}
{{- end }}

{{/*
Generate OpenTelemetry headers
*/}}
{{- define "genops-ai.otelHeaders" -}}
{{- $headers := list }}
{{- range $key, $value := .Values.opentelemetry.headers }}
{{- if $value }}
{{- $headers = append $headers (printf "%s=%s" $key $value) }}
{{- end }}
{{- end }}
{{- join "," $headers }}
{{- end }}

{{/*
Generate governance attributes as environment variables
*/}}
{{- define "genops-ai.governanceEnvVars" -}}
- name: GENOPS_TEAM
  value: {{ .Values.governance.defaultAttributes.team | quote }}
- name: GENOPS_PROJECT  
  value: {{ .Values.governance.defaultAttributes.project | quote }}
- name: GENOPS_ENVIRONMENT
  value: {{ .Values.governance.defaultAttributes.environment | quote }}
- name: GENOPS_COST_CENTER
  value: {{ .Values.governance.defaultAttributes.costCenter | quote }}
{{- end }}

{{/*
Generate network policy selectors
*/}}
{{- define "genops-ai.networkPolicySelectors" -}}
podSelector:
  matchLabels:
    {{- include "genops-ai.selectorLabels" . | nindent 4 }}
{{- end }}

{{/*
Validate configuration
*/}}
{{- define "genops-ai.validateConfig" -}}
{{- if and .Values.autoscaling.enabled (le (.Values.autoscaling.minReplicas | int) 0) }}
{{- fail "autoscaling.minReplicas must be greater than 0" }}
{{- end }}
{{- if and .Values.autoscaling.enabled (gt (.Values.autoscaling.minReplicas | int) (.Values.autoscaling.maxReplicas | int)) }}
{{- fail "autoscaling.minReplicas cannot be greater than autoscaling.maxReplicas" }}
{{- end }}
{{- if and .Values.providers.openai.enabled (not .Values.secrets.apiKeys.openai) (not .Values.providers.openai.apiKeySecret.name) }}
{{- fail "OpenAI provider is enabled but no API key configuration found" }}
{{- end }}
{{- if and .Values.providers.anthropic.enabled (not .Values.secrets.apiKeys.anthropic) (not .Values.providers.anthropic.apiKeySecret.name) }}
{{- fail "Anthropic provider is enabled but no API key configuration found" }}
{{- end }}
{{- if and .Values.providers.openrouter.enabled (not .Values.secrets.apiKeys.openrouter) (not .Values.providers.openrouter.apiKeySecret.name) }}
{{- fail "OpenRouter provider is enabled but no API key configuration found" }}
{{- end }}
{{- end }}

{{/*
Generate probe configuration
*/}}
{{- define "genops-ai.probeConfig" -}}
httpGet:
  path: {{ .path }}
  port: {{ .port | default 8000 }}
  scheme: {{ .scheme | default "HTTP" }}
initialDelaySeconds: {{ .initialDelaySeconds }}
periodSeconds: {{ .periodSeconds }}
timeoutSeconds: {{ .timeoutSeconds }}
failureThreshold: {{ .failureThreshold }}
successThreshold: {{ .successThreshold }}
{{- end }}

{{/*
Generate security context
*/}}
{{- define "genops-ai.securityContext" -}}
runAsNonRoot: true
runAsUser: 1000
runAsGroup: 1000
fsGroup: 1000
seccompProfile:
  type: RuntimeDefault
{{- end }}

{{/*
Generate container security context
*/}}
{{- define "genops-ai.containerSecurityContext" -}}
allowPrivilegeEscalation: false
readOnlyRootFilesystem: true
runAsNonRoot: true
runAsUser: 1000
runAsGroup: 1000
capabilities:
  drop:
  - ALL
seccompProfile:
  type: RuntimeDefault
{{- end }}

{{/*
Generate anti-affinity rules for high availability
*/}}
{{- define "genops-ai.antiAffinity" -}}
podAntiAffinity:
  preferredDuringSchedulingIgnoredDuringExecution:
  - weight: 100
    podAffinityTerm:
      labelSelector:
        matchExpressions:
        - key: app.kubernetes.io/name
          operator: In
          values:
          - {{ include "genops-ai.name" . }}
      topologyKey: kubernetes.io/hostname
  - weight: 50
    podAffinityTerm:
      labelSelector:
        matchExpressions:
        - key: app.kubernetes.io/name
          operator: In
          values:
          - {{ include "genops-ai.name" . }}
      topologyKey: topology.kubernetes.io/zone
{{- end }}