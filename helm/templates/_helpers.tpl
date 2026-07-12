{{- define "heart-disease-api.name" -}}
{{ .Chart.Name }}
{{- end -}}
{{- define "heart-disease-api.fullname" -}}
{{ .Release.Name }}-{{ .Chart.Name }}
{{- end -}}