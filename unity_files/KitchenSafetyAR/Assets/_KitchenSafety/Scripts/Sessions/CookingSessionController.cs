using System;
using System.Collections;

using KitchenSafety.Capture;
using KitchenSafety.Models;
using KitchenSafety.Networking;
using KitchenSafety.Scanning;

using TMPro;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

namespace KitchenSafety.Sessions
{
    /// <summary>
    /// Controls one sampled-frame cooking session.
    /// No video is recorded.
    /// </summary>
    public class CookingSessionController : MonoBehaviour
    {
        private enum LocalSessionState
        {
            None,
            Active,
            Paused,
            Stopped
        }


        [Header("Backend")]

        [SerializeField]
        private BackendClient backendClient;


        [Header("Existing Analysis Components")]

        [SerializeField]
        private ARCameraFrameUploader frameUploader;

        [SerializeField]
        private PeriodicScanController periodicScanController;


        [Header("Session Controls")]

        [SerializeField]
        private Button startSessionButton;

        [SerializeField]
        private Button pauseResumeButton;

        [SerializeField]
        private TMP_Text pauseResumeButtonLabel;

        [SerializeField]
        private Button stopSessionButton;

        [SerializeField]
        private Button analyseNowButton;

        [SerializeField]
        private TMP_Text sessionStatusText;


        [Header("Session Summary")]

        [SerializeField]
        private GameObject sessionSummaryPanel;

        [SerializeField]
        private TMP_Text summaryTitleText;

        [SerializeField]
        private TMP_Text presentHazardsText;

        [SerializeField]
        private TMP_Text possibleHazardsText;

        [SerializeField]
        private Button closeSummaryButton;


        [Header("Request")]

        [SerializeField]
        [Min(1)]
        private int timeoutSeconds = 30;


        private LocalSessionState localState =
            LocalSessionState.None;

        private string currentSessionId = "";

        private bool controlRequestInProgress;


        public bool HasActiveSession
        {
            get
            {
                return localState
                    == LocalSessionState.Active
                    && !string.IsNullOrWhiteSpace(
                        currentSessionId
                    );
            }
        }


        public bool HasOpenSession
        {
            get
            {
                return (
                    localState
                    == LocalSessionState.Active
                    || localState
                    == LocalSessionState.Paused
                )
                && !string.IsNullOrWhiteSpace(
                    currentSessionId
                );
            }
        }


        public string CurrentSessionId
        {
            get
            {
                return currentSessionId;
            }
        }


        private void Awake()
        {
            if (startSessionButton != null)
            {
                startSessionButton.onClick.AddListener(
                    StartCookingSession
                );
            }

            if (pauseResumeButton != null)
            {
                pauseResumeButton.onClick.AddListener(
                    TogglePauseResume
                );
            }

            if (stopSessionButton != null)
            {
                stopSessionButton.onClick.AddListener(
                    StopCookingSession
                );
            }

            if (closeSummaryButton != null)
            {
                closeSummaryButton.onClick.AddListener(
                    HideSummary
                );
            }
        }


        private void Start()
        {
            HideSummary();
            UpdateInterface();
            SetStatus("No cooking session");
        }


        private void OnDestroy()
        {
            if (startSessionButton != null)
            {
                startSessionButton.onClick.RemoveListener(
                    StartCookingSession
                );
            }

            if (pauseResumeButton != null)
            {
                pauseResumeButton.onClick.RemoveListener(
                    TogglePauseResume
                );
            }

            if (stopSessionButton != null)
            {
                stopSessionButton.onClick.RemoveListener(
                    StopCookingSession
                );
            }

            if (closeSummaryButton != null)
            {
                closeSummaryButton.onClick.RemoveListener(
                    HideSummary
                );
            }
        }


        public string GetAnalysisEndpoint()
        {
            if (!HasActiveSession)
            {
                return "/api/v1/live/analyse-image";
            }

            return (
                "/api/v1/sessions/"
                + currentSessionId
                + "/analyse-image"
            );
        }


        public void StartCookingSession()
        {
            if (
                controlRequestInProgress
                || HasOpenSession
            )
            {
                return;
            }

            StartCoroutine(
                StartSessionCoroutine()
            );
        }


        public void TogglePauseResume()
        {
            if (
                controlRequestInProgress
                || !HasOpenSession
            )
            {
                return;
            }

            if (
                localState
                == LocalSessionState.Active
            )
            {
                StartCoroutine(
                    PauseSessionCoroutine()
                );
            }
            else
            {
                StartCoroutine(
                    ResumeSessionCoroutine()
                );
            }
        }


        public void StopCookingSession()
        {
            if (
                controlRequestInProgress
                || !HasOpenSession
            )
            {
                return;
            }

            if (
                frameUploader != null
                && frameUploader.RequestInProgress
            )
            {
                SetStatus(
                    "Wait for the current analysis to finish."
                );
                return;
            }

            StartCoroutine(
                StopSessionCoroutine()
            );
        }


        private IEnumerator StartSessionCoroutine()
        {
            controlRequestInProgress = true;
            UpdateInterface();
            SetStatus("Starting cooking session…");

            string url = backendClient.BuildUrl(
                "/api/v1/sessions/start"
            );

            using UnityWebRequest request =
                CreatePostRequest(url);

            yield return request.SendWebRequest();

            if (!RequestSucceeded(request))
            {
                HandleFailure(
                    request,
                    "Could not start session."
                );
                FinishControlRequest();
                yield break;
            }

            SessionStartResponse response =
                JsonUtility.FromJson<
                    SessionStartResponse
                >(
                    request.downloadHandler.text
                );

            if (
                response == null
                || string.IsNullOrWhiteSpace(
                    response.session_id
                )
            )
            {
                SetStatus(
                    "Backend returned an invalid session."
                );
                FinishControlRequest();
                yield break;
            }

            currentSessionId = response.session_id;
            localState = LocalSessionState.Active;

            HideSummary();

            periodicScanController?.StartScanning();

            SetStatus("Session active · scanning");
            FinishControlRequest();
        }


        private IEnumerator PauseSessionCoroutine()
        {
            controlRequestInProgress = true;
            UpdateInterface();

            periodicScanController?.StopScanning();

            SetStatus("Pausing session…");

            string url = backendClient.BuildUrl(
                "/api/v1/sessions/"
                + currentSessionId
                + "/pause"
            );

            using UnityWebRequest request =
                CreatePostRequest(url);

            yield return request.SendWebRequest();

            if (!RequestSucceeded(request))
            {
                HandleFailure(
                    request,
                    "Could not pause session."
                );

                periodicScanController?.StartScanning();

                FinishControlRequest();
                yield break;
            }

            localState = LocalSessionState.Paused;
            SetStatus("Session paused");
            FinishControlRequest();
        }


        private IEnumerator ResumeSessionCoroutine()
        {
            controlRequestInProgress = true;
            UpdateInterface();
            SetStatus("Resuming session…");

            string url = backendClient.BuildUrl(
                "/api/v1/sessions/"
                + currentSessionId
                + "/resume"
            );

            using UnityWebRequest request =
                CreatePostRequest(url);

            yield return request.SendWebRequest();

            if (!RequestSucceeded(request))
            {
                HandleFailure(
                    request,
                    "Could not resume session."
                );
                FinishControlRequest();
                yield break;
            }

            localState = LocalSessionState.Active;

            periodicScanController?.StartScanning();

            SetStatus("Session active · scanning");
            FinishControlRequest();
        }


        private IEnumerator StopSessionCoroutine()
        {
            controlRequestInProgress = true;
            UpdateInterface();

            periodicScanController?.StopScanning();

            SetStatus("Finalising session…");

            string sessionIdAtStop =
                currentSessionId;

            string url = backendClient.BuildUrl(
                "/api/v1/sessions/"
                + sessionIdAtStop
                + "/stop"
            );

            using UnityWebRequest request =
                CreatePostRequest(url);

            yield return request.SendWebRequest();

            if (!RequestSucceeded(request))
            {
                HandleFailure(
                    request,
                    "Could not stop session."
                );
                FinishControlRequest();
                yield break;
            }

            SessionSummaryResponse summary =
                JsonUtility.FromJson<
                    SessionSummaryResponse
                >(
                    request.downloadHandler.text
                );

            localState = LocalSessionState.Stopped;
            currentSessionId = "";

            DisplaySummary(summary);

            SetStatus("Session completed");
            FinishControlRequest();
        }


        private UnityWebRequest CreatePostRequest(
            string url
        )
        {
            UnityWebRequest request =
                new UnityWebRequest(
                    url,
                    UnityWebRequest.kHttpVerbPOST
                );

            request.uploadHandler =
                new UploadHandlerRaw(
                    Array.Empty<byte>()
                );

            request.downloadHandler =
                new DownloadHandlerBuffer();

            request.SetRequestHeader(
                "Content-Type",
                "application/json"
            );

            request.timeout = timeoutSeconds;

            return request;
        }


        private static bool RequestSucceeded(
            UnityWebRequest request
        )
        {
            return request.result
                == UnityWebRequest.Result.Success;
        }


        private void HandleFailure(
            UnityWebRequest request,
            string fallbackMessage
        )
        {
            string responseBody =
                request.downloadHandler != null
                    ? request.downloadHandler.text
                    : "";

            Debug.LogError(
                fallbackMessage
                + "\nHTTP: "
                + request.responseCode
                + "\nError: "
                + request.error
                + "\nResponse: "
                + responseBody
            );

            SetStatus(
                fallbackMessage
                + " "
                + request.error
            );
        }


        private void DisplaySummary(
            SessionSummaryResponse summary
        )
        {
            if (summary == null)
            {
                return;
            }

            if (summaryTitleText != null)
            {
                summaryTitleText.text =
                    "Cooking Session Summary\n"
                    + summary.analysed_frame_count
                    + " frames analysed";
            }

            if (presentHazardsText != null)
            {
                presentHazardsText.text =
                    BuildSummarySection(
                        "PRESENT HAZARDS",
                        summary.present_hazards
                    );
            }

            if (possibleHazardsText != null)
            {
                possibleHazardsText.text =
                    BuildSummarySection(
                        "POSSIBLE HAZARDS",
                        summary.possible_hazards
                    );
            }

            if (sessionSummaryPanel != null)
            {
                sessionSummaryPanel.transform.SetAsLastSibling();
                sessionSummaryPanel.SetActive(true);
            }
        }


        private static string BuildSummarySection(
            string heading,
            string[] hazardNames
        )
        {
            if (
                hazardNames == null
                || hazardNames.Length == 0
            )
            {
                return heading + "\nNone";
            }

            return (
                heading
                + "\n• "
                + string.Join(
                    "\n• ",
                    hazardNames
                )
            );
        }


        public void HideSummary()
        {
            sessionSummaryPanel?.SetActive(false);
        }


        private void FinishControlRequest()
        {
            controlRequestInProgress = false;
            UpdateInterface();
        }


        private void UpdateInterface()
        {
            bool active =
                localState
                == LocalSessionState.Active;

            bool paused =
                localState
                == LocalSessionState.Paused;

            bool open = active || paused;

            if (startSessionButton != null)
            {
                startSessionButton.interactable =
                    !open
                    && !controlRequestInProgress;
            }

            if (pauseResumeButton != null)
            {
                pauseResumeButton.interactable =
                    open
                    && !controlRequestInProgress;
            }

            if (stopSessionButton != null)
            {
                stopSessionButton.interactable =
                    open
                    && !controlRequestInProgress;
            }

            if (analyseNowButton != null)
            {
                analyseNowButton.interactable =
                    !paused
                    && !controlRequestInProgress;
            }

            if (pauseResumeButtonLabel != null)
            {
                pauseResumeButtonLabel.text =
                    paused
                        ? "Resume Session"
                        : "Pause Session";
            }
        }


        private void SetStatus(
            string message
        )
        {
            if (sessionStatusText != null)
            {
                sessionStatusText.text = message;
            }

            Debug.Log(message);
        }
    }
}
