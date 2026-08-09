using System.Collections;
using System.Collections.Generic;

using KitchenSafety.Models;

using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace KitchenSafety.UI
{
    /// <summary>
    /// Displays up to three concise, scrollable hazard cards.
    /// </summary>
    public class HazardDisplay : MonoBehaviour
    {
        [Header("Panel")]

        [SerializeField]
        private GameObject hazardResultsPanel;


        [Header("Header")]

        [SerializeField]
        private TMP_Text resultsHeaderText;


        [Header("Scrollable Cards")]

        [SerializeField]
        private ScrollRect hazardScrollRect;

        [SerializeField]
        private RectTransform hazardCardsContent;


        [Header("Localisation")]

        [SerializeField]
        private HazardBoxOverlay hazardBoxOverlay;


        [Header("Display Timing")]

        [SerializeField]
        [Min(1f)]
        private float minimumDisplaySeconds = 20f;


        [Header("Card Settings")]

        [SerializeField]
        [Range(1, 3)]
        private int maximumDisplayedHazards = 3;

        [SerializeField]
        [Min(140f)]
        private float cardHeight = 195f;

        [SerializeField]
        [Min(16f)]
        private float titleFontSize = 25f;

        [SerializeField]
        [Min(14f)]
        private float bodyFontSize = 21f;


        private readonly List<GameObject> activeCards =
            new List<GameObject>();

        private Coroutine hideCoroutine;
        private float warningDisplayedAtRealtime;
        private int warningVersion;


        private void Start()
        {
            HideHazardImmediate();
        }


        private void OnDisable()
        {
            StopHideCoroutine();
        }


        public void DisplayResponse(
            HazardAnalysisResponse response
        )
        {
            if (response == null)
            {
                Debug.LogWarning(
                    "HazardDisplay received a null response."
                );
                return;
            }

            bool containsHazards =
                response.hazards_present
                && response.hazards != null
                && response.hazards.Length > 0;

            if (!containsHazards)
            {
                HandleSafeResponse();
                return;
            }

            DisplayHazards(response);
        }


        public void NotifyCaptureStarted()
        {
            hazardBoxOverlay?.ClearBoxes();
        }


        private void DisplayHazards(
            HazardAnalysisResponse response
        )
        {
            warningVersion++;

            StopHideCoroutine();
            ClearCards();

            int hazardCount = Mathf.Min(
                response.hazards.Length,
                maximumDisplayedHazards
            );

            if (resultsHeaderText != null)
            {
                resultsHeaderText.text =
                    hazardCount == 1
                        ? "1 hazard detected"
                        : hazardCount
                          + " hazards detected";
            }

            for (
                int index = 0;
                index < hazardCount;
                index++
            )
            {
                CreateHazardCard(
                    response.hazards[index],
                    index + 1
                );
            }

            hazardResultsPanel?.SetActive(true);

            hazardBoxOverlay?.DisplayResponse(
                response
            );

            Canvas.ForceUpdateCanvases();

            if (hazardScrollRect != null)
            {
                hazardScrollRect
                    .verticalNormalizedPosition = 1f;
            }

            warningDisplayedAtRealtime =
                Time.realtimeSinceStartup;

            int currentVersion = warningVersion;

            hideCoroutine = StartCoroutine(
                HideAfterMinimumDuration(
                    currentVersion,
                    minimumDisplaySeconds
                )
            );
        }


        private void CreateHazardCard(
            HazardItem hazard,
            int displayIndex
        )
        {
            GameObject cardObject = new GameObject(
                $"Hazard Card {displayIndex}",
                typeof(RectTransform),
                typeof(Image),
                typeof(LayoutElement),
                typeof(VerticalLayoutGroup)
            );

            RectTransform cardRect =
                cardObject.GetComponent<RectTransform>();

            cardRect.SetParent(
                hazardCardsContent,
                false
            );

            Image cardImage =
                cardObject.GetComponent<Image>();

            cardImage.color = GetCardColour(
                hazard.severity
            );

            cardImage.raycastTarget = false;

            LayoutElement cardLayout =
                cardObject.GetComponent<LayoutElement>();

            bool isPossible =
                string.Equals(
                    hazard.hazard_status,
                    "possible",
                    System.StringComparison
                        .OrdinalIgnoreCase
                );

            float effectiveCardHeight =
                isPossible
                    ? cardHeight + 28f
                    : cardHeight;

            cardLayout.minHeight =
                effectiveCardHeight;
            cardLayout.preferredHeight =
                effectiveCardHeight;
            cardLayout.flexibleHeight = 0f;

            VerticalLayoutGroup layout =
                cardObject.GetComponent<
                    VerticalLayoutGroup
                >();

            layout.padding =
                new RectOffset(
                    16,
                    16,
                    12,
                    12
                );

            layout.spacing = 5f;
            layout.childAlignment =
                TextAnchor.UpperLeft;
            layout.childControlWidth = true;
            layout.childControlHeight = true;
            layout.childForceExpandWidth = true;
            layout.childForceExpandHeight = false;

            string severity =
                string.IsNullOrWhiteSpace(
                    hazard.severity
                )
                    ? "HAZARD"
                    : hazard.severity
                        .ToUpperInvariant();

            string status =
                isPossible
                    ? "POSSIBLE"
                    : "PRESENT";

            CreateCardText(
                cardRect,
                "Hazard Title",
                displayIndex
                + ". "
                + status
                + " · "
                + severity
                + " · "
                + hazard.description,
                titleFontSize,
                FontStyles.Bold,
                54f
            );

            CreateCardText(
                cardRect,
                "Hazard Reason",
                hazard.explanation,
                bodyFontSize,
                FontStyles.Normal,
                42f
            );

            string actionText =
                "Action: "
                + hazard.recommended_action;

            if (
                isPossible
                && !string.IsNullOrWhiteSpace(
                    hazard.trigger_condition
                )
            )
            {
                actionText += (
                    "\nIf: "
                    + hazard.trigger_condition
                );
            }

            CreateCardText(
                cardRect,
                "Hazard Action",
                actionText,
                bodyFontSize,
                FontStyles.Bold,
                isPossible
                    ? 72f
                    : 48f
            );

            activeCards.Add(cardObject);
        }


        private static void CreateCardText(
            RectTransform parent,
            string objectName,
            string text,
            float fontSize,
            FontStyles fontStyle,
            float preferredHeight
        )
        {
            GameObject textObject = new GameObject(
                objectName,
                typeof(RectTransform),
                typeof(TextMeshProUGUI),
                typeof(LayoutElement)
            );

            RectTransform textRect =
                textObject.GetComponent<RectTransform>();

            textRect.SetParent(
                parent,
                false
            );

            TextMeshProUGUI textComponent =
                textObject.GetComponent<TextMeshProUGUI>();

            textComponent.text = text ?? "";
            textComponent.fontSize = fontSize;
            textComponent.fontStyle = fontStyle;
            textComponent.alignment =
                TextAlignmentOptions.TopLeft;
            textComponent.color = Color.white;
            textComponent.textWrappingMode =
                TextWrappingModes.Normal;
            textComponent.overflowMode =
                TextOverflowModes.Ellipsis;
            textComponent.raycastTarget = false;

            LayoutElement textLayout =
                textObject.GetComponent<LayoutElement>();

            textLayout.minHeight = preferredHeight;
            textLayout.preferredHeight = preferredHeight;
            textLayout.flexibleHeight = 0f;
        }


        private void HandleSafeResponse()
        {
            if (
                hazardResultsPanel == null
                || !hazardResultsPanel.activeSelf
            )
            {
                hazardBoxOverlay?.ClearBoxes();
                return;
            }

            float elapsed =
                Time.realtimeSinceStartup
                - warningDisplayedAtRealtime;

            if (elapsed >= minimumDisplaySeconds)
            {
                HideHazardImmediate();
                return;
            }

            StopHideCoroutine();

            hideCoroutine = StartCoroutine(
                HideAfterMinimumDuration(
                    warningVersion,
                    minimumDisplaySeconds - elapsed
                )
            );
        }


        private IEnumerator HideAfterMinimumDuration(
            int version,
            float delaySeconds
        )
        {
            yield return new WaitForSecondsRealtime(
                delaySeconds
            );

            if (version != warningVersion)
            {
                yield break;
            }

            HideHazardImmediate();
        }


        private void StopHideCoroutine()
        {
            if (hideCoroutine == null)
            {
                return;
            }

            StopCoroutine(hideCoroutine);
            hideCoroutine = null;
        }


        private void ClearCards()
        {
            foreach (
                GameObject card
                in activeCards
            )
            {
                if (card != null)
                {
                    Destroy(card);
                }
            }

            activeCards.Clear();
        }


        public void HideHazardImmediate()
        {
            StopHideCoroutine();
            ClearCards();
            hazardBoxOverlay?.ClearBoxes();
            hazardResultsPanel?.SetActive(false);
        }


        public void HideHazard()
        {
            HideHazardImmediate();
        }


        private static Color GetCardColour(
            string severity
        )
        {
            switch (
                severity
                    ?.Trim()
                    .ToLowerInvariant()
            )
            {
                case "low":
                    return new Color(
                        0.48f,
                        0.38f,
                        0.04f,
                        0.94f
                    );

                case "medium":
                    return new Color(
                        0.62f,
                        0.26f,
                        0.03f,
                        0.94f
                    );

                case "high":
                    return new Color(
                        0.58f,
                        0.08f,
                        0.05f,
                        0.95f
                    );

                case "critical":
                    return new Color(
                        0.43f,
                        0.02f,
                        0.16f,
                        0.96f
                    );

                default:
                    return new Color(
                        0.05f,
                        0.12f,
                        0.27f,
                        0.94f
                    );
            }
        }
    }
}
