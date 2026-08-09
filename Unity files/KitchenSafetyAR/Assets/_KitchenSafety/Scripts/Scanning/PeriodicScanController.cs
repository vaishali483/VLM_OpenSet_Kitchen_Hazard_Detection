using System.Collections;

using KitchenSafety.Capture;

using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace KitchenSafety.Scanning
{
    /// <summary>
    /// Controls periodic near-real-time scanning.
    ///
    /// A new frame is captured only after the previous
    /// request has fully completed.
    /// </summary>
    public class PeriodicScanController : MonoBehaviour
    {
        [Header("Capture")]

        [SerializeField]
        private ARCameraFrameUploader frameUploader;


        [Header("Scanning")]

        [Tooltip(
            "Time to wait after one analysis finishes " +
            "before starting the next capture."
        )]
        [SerializeField]
        [Min(1f)]
        private float scanIntervalSeconds = 5f;

        [Tooltip(
            "Wait briefly before retrying when no camera " +
            "CPU image is available."
        )]
        [SerializeField]
        [Min(0.1f)]
        private float cameraRetryDelaySeconds = 0.5f;


        [Header("Interface")]

        [SerializeField]
        private Button scanningToggleButton;

        [SerializeField]
        private TMP_Text scanningButtonLabel;

        [SerializeField]
        private Button analyseNowButton;

        [SerializeField]
        private TMP_Text captureStatusText;


        private bool scanningActive;
        private Coroutine scanningCoroutine;


        public bool ScanningActive
        {
            get
            {
                return scanningActive;
            }
        }


        private void Awake()
        {
            if (scanningToggleButton != null)
            {
                scanningToggleButton.onClick.AddListener(
                    ToggleScanning
                );
            }
        }


        private void Start()
        {
            UpdateInterface();
        }


        private void OnDisable()
        {
            StopScanning();
        }


        private void OnDestroy()
        {
            if (scanningToggleButton != null)
            {
                scanningToggleButton.onClick.RemoveListener(
                    ToggleScanning
                );
            }
        }


        /// <summary>
        /// Switches between continuous periodic scanning
        /// and the paused state.
        /// </summary>
        public void ToggleScanning()
        {
            if (scanningActive)
            {
                StopScanning();
            }
            else
            {
                StartScanning();
            }
        }


        public void StartScanning()
        {
            if (scanningActive)
            {
                return;
            }

            if (frameUploader == null)
            {
                SetStatus(
                    "Frame uploader is not assigned."
                );

                return;
            }

            if (!frameUploader.IsReady)
            {
                SetStatus(
                    "Camera or backend is not ready."
                );

                return;
            }

            scanningActive = true;

            scanningCoroutine = StartCoroutine(
                ScanningLoop()
            );

            SetStatus(
                "Scanning started."
            );

            UpdateInterface();
        }


        public void StopScanning()
        {
            if (!scanningActive)
            {
                UpdateInterface();
                return;
            }

            scanningActive = false;

            if (scanningCoroutine != null)
            {
                StopCoroutine(scanningCoroutine);
                scanningCoroutine = null;
            }

            // Stopping scanning does not cancel a request that
            // is already uploading. That request may complete
            // normally, but no new request will be started.
            SetStatus(
                frameUploader != null
                && frameUploader.RequestInProgress
                    ? "Scanning paused. Finishing current analysis…"
                    : "Scanning paused."
            );

            UpdateInterface();
        }


        private IEnumerator ScanningLoop()
        {
            while (scanningActive)
            {
                // Never start another capture while the
                // previous request is running.
                if (frameUploader.RequestInProgress)
                {
                    yield return null;
                    continue;
                }

                bool captureStarted =
                    frameUploader.TryStartCaptureAndUpload();

                if (!captureStarted)
                {
                    // ARCore can occasionally have no CPU image
                    // ready at the exact requested moment.
                    yield return new WaitForSecondsRealtime(
                        cameraRetryDelaySeconds
                    );

                    continue;
                }

                // Wait for conversion, upload and response.
                yield return new WaitUntil(
                    () =>
                        frameUploader == null
                        || !frameUploader.RequestInProgress
                        || !scanningActive
                );

                if (!scanningActive)
                {
                    break;
                }

                SetStatus(
                    $"Next scan in " +
                    $"{scanIntervalSeconds:0} seconds…"
                );

                yield return new WaitForSecondsRealtime(
                    scanIntervalSeconds
                );
            }

            scanningCoroutine = null;
        }


        private void UpdateInterface()
        {
            if (scanningButtonLabel != null)
            {
                scanningButtonLabel.text =
                    scanningActive
                        ? "Pause Scanning"
                        : "Start Scanning";
            }

            if (analyseNowButton != null)
            {
                // Avoid manual requests while automatic
                // scanning is enabled.
                analyseNowButton.interactable =
                    !scanningActive;
            }
        }


        private void SetStatus(string message)
        {
            if (captureStatusText != null)
            {
                captureStatusText.text = message;
            }

            Debug.Log(message);
        }
    }
}