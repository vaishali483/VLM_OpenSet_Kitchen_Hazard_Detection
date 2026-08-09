using System.Collections;
using KitchenSafety.Models;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

namespace KitchenSafety.Networking
{
    /// <summary>
    /// Tests communication between Unity and the FastAPI backend.
    /// </summary>
    public class BackendClient : MonoBehaviour
    {
        [Header("Backend")]

        [Tooltip(
            "FastAPI base URL using the computer's local IPv4 address."
        )]
        [SerializeField]
        private string baseUrl = "http://10.132.132.3:8000";

        [SerializeField]
        [Min(1)]
        private int timeoutSeconds = 10;

        [Header("Interface")]

        [SerializeField]
        private TMP_Text connectionStatusText;

        [SerializeField]
        private Button testBackendButton;

        private bool requestInProgress;

        private void Awake()
        {
            if (testBackendButton != null)
            {
                testBackendButton.onClick.AddListener(
                    TestConnection
                );
            }
        }

        private void Start()
        {
            TestConnection();
        }

        private void OnDestroy()
        {
            if (testBackendButton != null)
            {
                testBackendButton.onClick.RemoveListener(
                    TestConnection
                );
            }
        }

        /// <summary>
        /// Public method used by the Test button.
        /// </summary>
        public void TestConnection()
        {
            if (requestInProgress)
            {
                return;
            }

            StartCoroutine(CheckBackendHealth());
        }

        private IEnumerator CheckBackendHealth()
        {
            requestInProgress = true;

            SetStatus(
                "Backend: Connecting…"
            );

            if (testBackendButton != null)
            {
                testBackendButton.interactable = false;
            }

            string healthUrl =
                BuildUrl("/api/v1/health");

            using UnityWebRequest request =
                UnityWebRequest.Get(healthUrl);

            request.timeout = timeoutSeconds;

            yield return request.SendWebRequest();

            if (
                request.result
                != UnityWebRequest.Result.Success
            )
            {
                Debug.LogError(
                    $"Backend connection failed: " +
                    $"{request.error}\nURL: {healthUrl}"
                );

                SetStatus(
                    $"Backend: Connection failed\n" +
                    $"{request.error}"
                );
            }
            else
            {
                HandleSuccessfulResponse(
                    request.downloadHandler.text
                );
            }

            requestInProgress = false;

            if (testBackendButton != null)
            {
                testBackendButton.interactable = true;
            }
        }

        private void HandleSuccessfulResponse(
            string responseJson
        )
        {
            try
            {
                HealthResponse response =
                    JsonUtility.FromJson<HealthResponse>(
                        responseJson
                    );

                if (
                    response == null
                    || response.status != "ok"
                )
                {
                    SetStatus(
                        "Backend: Invalid health response"
                    );

                    Debug.LogWarning(
                        $"Unexpected backend response: " +
                        $"{responseJson}"
                    );

                    return;
                }

                SetStatus(
                    $"Backend: Connected\n" +
                    $"{response.service} v{response.version}"
                );

                Debug.Log(
                    $"Backend connected successfully.\n" +
                    $"{responseJson}"
                );
            }
            catch (System.Exception exception)
            {
                SetStatus(
                    "Backend: Invalid JSON response"
                );

                Debug.LogError(
                    $"Could not parse health response: " +
                    $"{exception.Message}\n" +
                    $"Response: {responseJson}"
                );
            }
        }

        private void SetStatus(string message)
        {
            if (connectionStatusText != null)
            {
                connectionStatusText.text = message;
            }

            Debug.Log(message);
        }

        public string BaseUrl
        {
            get
            {
                return baseUrl.TrimEnd('/');
            }
        }


        public string BuildUrl(string relativePath)
        {
            return $"{BaseUrl}/{relativePath.TrimStart('/')}";
        }
    }
}