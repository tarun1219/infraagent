# Kubernetes Ingress Patterns

## Overview
Ingress exposes HTTP/HTTPS routes from outside the cluster to Services. An Ingress Controller (nginx, Traefik, AWS ALB, Contour) implements the rules. The `networking.k8s.io/v1` API is stable since Kubernetes 1.19.

## Basic Ingress with TLS

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-app-ingress
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.example.com
      secretName: api-example-com-tls    # cert-manager will populate this
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: my-app
                port:
                  number: 80
```

## Path Types

| pathType | Behavior |
|----------|----------|
| `Exact` | Matches exactly `/foo` but not `/foo/` |
| `Prefix` | Matches `/foo`, `/foo/`, `/foo/bar` |
| `ImplementationSpecific` | Behavior depends on IngressClass |

## Multi-Service Ingress

```yaml
spec:
  rules:
    - host: example.com
      http:
        paths:
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: api-service
                port:
                  number: 8080
          - path: /static
            pathType: Prefix
            backend:
              service:
                name: static-service
                port:
                  number: 80
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend-service
                port:
                  number: 3000
```

## AWS ALB Ingress (AWS Load Balancer Controller)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-app-alb
  namespace: production
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:us-east-1:123456789012:certificate/abc
    alb.ingress.kubernetes.io/ssl-policy: ELBSecurityPolicy-TLS13-1-2-2021-06
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP":80},{"HTTPS":443}]'
    alb.ingress.kubernetes.io/ssl-redirect: "443"
spec:
  ingressClassName: alb
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: my-app
                port:
                  number: 80
```

## cert-manager ClusterIssuer

```yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: ops@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
      - http01:
          ingress:
            class: nginx
```

## Common Mistakes
- Using deprecated `extensions/v1beta1` or `networking.k8s.io/v1beta1` APIs (removed in K8s 1.22)
- No `ingressClassName` — without it, the Ingress may not be picked up by any controller
- Missing TLS secret — cert-manager needs time to issue the certificate; app may briefly serve HTTP only
- Overlapping path rules with incorrect order — more specific paths should come before catch-all `/`
