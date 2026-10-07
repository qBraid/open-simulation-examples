# Connect IBM Quantum to qBraid (5 minutes)

You need two things: a free qBraid account and a free IBM Quantum account.

## 1. IBM Quantum: get an API key and an instance CRN

1. Sign in at https://quantum.cloud.ibm.com with an IBMid or IBM Cloud login.
2. **API key:** on the dashboard, create an API key and copy it. Keep it private.
3. **Instance CRN:** open **Instances**, hover over your instance's CRN and
   click copy. It starts with `crn:v1:bluemix:public:quantum-computing:us-east:`.
   If you have no instance, create one on the free **Open Plan** (us-east region).

The Open Plan gives up to 10 QPU minutes per rolling 28 days. Sessions are not
available on it; jobs and batches are.

## 2. qBraid: save them in the Vault

1. Sign in at https://account.qbraid.com and launch Lab.
2. Click your profile icon (top right), then **Vault**, then **Add Key**.
3. Provider **IBM Quantum**. Paste the API key and the instance CRN.
   Channel: **IBM Quantum Platform**. Leave the URL as `https://cloud.ibm.com`.
4. Save.

Prefer to be guided click by click? In a Lab terminal run `qbraid-ui tour submit-ibm-job`.

## 3. Install the webinar environment

```
qbraid envs install qiskit_zngl3z
```

Or use the Environment Manager in the Lab sidebar. Then pick its kernel for your notebook.

## 4. Check it works

```python
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, min_num_qubits=4)
print("Connected. Least busy QPU:", backend.name)
```

This only lists backends; it uses no QPU time.

## Troubleshooting

| You see | Fix |
|---|---|
| `AccountNotFoundError: Unable to find account` | The key isn't saved in this Lab instance. Add it in the Vault (step 2). |
| Empty backend list, or a warning `Invalid instance crn:...` | The API key or CRN is wrong. Copy both again and re-save in the Vault. |
| `401 ... Error authenticating user` | The API key was deleted or mistyped. Create a new one and update the Vault. |
| Connection error mentioning `quantum.quantum.cloud.ibm.com` | Set the Vault URL back to `https://cloud.ibm.com`. |
| "Session" refused | Open Plan has no sessions. Use job or batch mode. |
| Long queue | Normal for busy QPUs. `least_busy` picks the shortest queue. |

IBM bills IBM jobs to your IBM plan, not your qBraid credits. Your jobs appear
in qBraid's Quantum Jobs panel under the **IBM Cloud** tab, and on your IBM dashboard.
