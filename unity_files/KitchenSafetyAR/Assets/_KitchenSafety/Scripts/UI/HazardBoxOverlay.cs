using System.Collections.Generic;

using KitchenSafety.Models;

using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace KitchenSafety.UI
{
    /// <summary>
    /// Displays validated hazard boxes over the last analysed
    /// captured image.
    /// </summary>
    public class HazardBoxOverlay : MonoBehaviour
    {
        [Header("Layer")]

        [SerializeField]
        private RectTransform boxLayer;


        [Header("Appearance")]

        [SerializeField]
        [Min(1f)]
        private float borderThickness = 5f;

        [SerializeField]
        private bool showLabels = true;

        [SerializeField]
        [Min(16f)]
        private float labelFontSize = 24f;


        private readonly List<GameObject> activeBoxes =
            new List<GameObject>();


        private void Awake()
        {
            if (boxLayer == null)
            {
                boxLayer = transform as RectTransform;
            }
        }


        public void DisplayResponse(
            HazardAnalysisResponse response
        )
        {
            ClearBoxes();

            if (
                response == null
                || response.hazards == null
            )
            {
                return;
            }

            for (
                int index = 0;
                index < response.hazards.Length;
                index++
            )
            {
                HazardItem hazard =
                    response.hazards[index];

                if (!CanDisplayBoundingBox(hazard))
                {
                    continue;
                }

                // Use the real hazard number so box number 2
                // always matches hazard card number 2.
                CreateHazardBox(
                    hazard,
                    index + 1
                );
            }
        }


        private static bool CanDisplayBoundingBox(
            HazardItem hazard
        )
        {
            if (
                hazard == null
                || hazard.visualisation == null
            )
            {
                return false;
            }

            string visualisationMode =
                hazard.visualisation.mode;

            bool isExactBox =
                visualisationMode == "bounding_box";

            bool isApproximateBox =
                visualisationMode == "approximate_box";

            if (!isExactBox && !isApproximateBox)
            {
                return false;
            }

            NormalizedBoundingBox box =
                hazard.visualisation.bbox_normalized;

            if (box == null)
            {
                return false;
            }

            float xMin = Mathf.Clamp01(
                box.x_min
            );

            float xMax = Mathf.Clamp01(
                box.x_max
            );

            float yMin = Mathf.Clamp01(
                box.y_min
            );

            float yMax = Mathf.Clamp01(
                box.y_max
            );

            return xMax > xMin
                && yMax > yMin;
        }


        private void CreateHazardBox(
            HazardItem hazard,
            int hazardNumber
        )
        {
            NormalizedBoundingBox box =
                hazard.visualisation.bbox_normalized;

            float xMin = Mathf.Clamp01(
                box.x_min
            );

            float xMax = Mathf.Clamp01(
                box.x_max
            );

            float yMin = Mathf.Clamp01(
                box.y_min
            );

            float yMax = Mathf.Clamp01(
                box.y_max
            );

            // Backend origin is top-left.
            // Unity UI origin is bottom-left.
            float unityYMin = 1f - yMax;
            float unityYMax = 1f - yMin;

            GameObject boxObject = new GameObject(
                $"Hazard Box {hazardNumber}",
                typeof(RectTransform)
            );

            RectTransform boxRect =
                boxObject.GetComponent<RectTransform>();

            boxRect.SetParent(
                boxLayer,
                worldPositionStays: false
            );

            boxRect.anchorMin = new Vector2(
                xMin,
                unityYMin
            );

            boxRect.anchorMax = new Vector2(
                xMax,
                unityYMax
            );

            boxRect.offsetMin = Vector2.zero;
            boxRect.offsetMax = Vector2.zero;

            bool isApproximate =
                hazard.visualisation.mode
                == "approximate_box";

            Color boxColour =
                GetRiskColour(
                    hazard.severity
                );

            if (isApproximate)
            {
                boxColour.a = 0.75f;
            }

            float effectiveBorderThickness =
                isApproximate
                    ? borderThickness * 0.65f
                    : borderThickness;

            CreateBorder(
                boxRect,
                "Top",
                BorderSide.Top,
                boxColour,
                effectiveBorderThickness
            );

            CreateBorder(
                boxRect,
                "Bottom",
                BorderSide.Bottom,
                boxColour,
                effectiveBorderThickness
            );

            CreateBorder(
                boxRect,
                "Left",
                BorderSide.Left,
                boxColour,
                effectiveBorderThickness
            );

            CreateBorder(
                boxRect,
                "Right",
                BorderSide.Right,
                boxColour,
                effectiveBorderThickness
            );

            if (showLabels)
            {
                CreateLabel(
                    boxRect,
                    hazard,
                    hazardNumber,
                    boxColour
                );
            }

            activeBoxes.Add(boxObject);
        }


        private enum BorderSide
        {
            Top,
            Bottom,
            Left,
            Right
        }


        private void CreateBorder(
            RectTransform parent,
            string objectName,
            BorderSide side,
            Color colour,
            float thickness
        )
        {
            GameObject borderObject = new GameObject(
                objectName,
                typeof(RectTransform),
                typeof(Image)
            );

            RectTransform borderRect =
                borderObject.GetComponent<RectTransform>();

            borderRect.SetParent(
                parent,
                worldPositionStays: false
            );

            Image borderImage =
                borderObject.GetComponent<Image>();

            borderImage.color = colour;
            borderImage.raycastTarget = false;

            switch (side)
            {
                case BorderSide.Top:
                    borderRect.anchorMin =
                        new Vector2(0f, 1f);

                    borderRect.anchorMax =
                        new Vector2(1f, 1f);

                    borderRect.pivot =
                        new Vector2(0.5f, 1f);

                    borderRect.sizeDelta =
                        new Vector2(
                            0f,
                            thickness
                        );
                    break;

                case BorderSide.Bottom:
                    borderRect.anchorMin =
                        new Vector2(0f, 0f);

                    borderRect.anchorMax =
                        new Vector2(1f, 0f);

                    borderRect.pivot =
                        new Vector2(0.5f, 0f);

                    borderRect.sizeDelta =
                        new Vector2(
                            0f,
                            thickness
                        );
                    break;

                case BorderSide.Left:
                    borderRect.anchorMin =
                        new Vector2(0f, 0f);

                    borderRect.anchorMax =
                        new Vector2(0f, 1f);

                    borderRect.pivot =
                        new Vector2(0f, 0.5f);

                    borderRect.sizeDelta =
                        new Vector2(
                            thickness,
                            0f
                        );
                    break;

                case BorderSide.Right:
                    borderRect.anchorMin =
                        new Vector2(1f, 0f);

                    borderRect.anchorMax =
                        new Vector2(1f, 1f);

                    borderRect.pivot =
                        new Vector2(1f, 0.5f);

                    borderRect.sizeDelta =
                        new Vector2(
                            thickness,
                            0f
                        );
                    break;
            }

            borderRect.anchoredPosition =
                Vector2.zero;
        }


        private void CreateLabel(
            RectTransform boxRect,
            HazardItem hazard,
            int hazardNumber,
            Color colour
        )
        {
            bool isApproximate =
                hazard.visualisation.mode
                == "approximate_box";

            GameObject labelBackground =
                new GameObject(
                    "Hazard Label",
                    typeof(RectTransform),
                    typeof(Image)
                );

            RectTransform labelRect =
                labelBackground
                    .GetComponent<RectTransform>();

            labelRect.SetParent(
                boxRect,
                worldPositionStays: false
            );

            labelRect.anchorMin =
                new Vector2(0f, 1f);

            labelRect.anchorMax =
                new Vector2(0f, 1f);

            labelRect.pivot =
                new Vector2(0f, 0f);

            labelRect.anchoredPosition =
                Vector2.zero;

            labelRect.sizeDelta =
                new Vector2(
                    260f,
                    44f
                );

            Image backgroundImage =
                labelBackground.GetComponent<Image>();

            Color backgroundColour = colour;
            backgroundColour.a =
                isApproximate
                    ? 0.78f
                    : 0.92f;

            backgroundImage.color =
                backgroundColour;

            backgroundImage.raycastTarget =
                false;

            GameObject textObject = new GameObject(
                "Label Text",
                typeof(RectTransform),
                typeof(TextMeshProUGUI)
            );

            RectTransform textRect =
                textObject.GetComponent<RectTransform>();

            textRect.SetParent(
                labelRect,
                worldPositionStays: false
            );

            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;

            textRect.offsetMin =
                new Vector2(6f, 2f);

            textRect.offsetMax =
                new Vector2(-6f, -2f);

            TextMeshProUGUI labelText =
                textObject.GetComponent<TextMeshProUGUI>();

            string severity =
                string.IsNullOrWhiteSpace(
                    hazard.severity
                )
                    ? "HAZARD"
                    : hazard.severity.ToUpperInvariant();

            string detectedClass =
                hazard.visualisation
                    .source_class_name;

            if (isApproximate)
            {
                labelText.text =
                    $"{hazardNumber}. APPROX. AREA";
            }
            else if (
                string.IsNullOrWhiteSpace(
                    detectedClass
                )
            )
            {
                labelText.text =
                    $"{hazardNumber}. {severity}";
            }
            else
            {
                labelText.text =
                    $"{hazardNumber}. {severity} · "
                    + detectedClass.ToUpperInvariant();
            }

            labelText.fontSize =
                labelFontSize;

            labelText.fontStyle =
                FontStyles.Bold;

            labelText.alignment =
                TextAlignmentOptions.MidlineLeft;

            labelText.color = Color.white;

            labelText.textWrappingMode =
                TextWrappingModes.NoWrap;

            labelText.overflowMode =
                TextOverflowModes.Ellipsis;

            labelText.raycastTarget = false;
        }


        public void ClearBoxes()
        {
            foreach (
                GameObject box
                in activeBoxes
            )
            {
                if (box != null)
                {
                    Destroy(box);
                }
            }

            activeBoxes.Clear();
        }


        private static Color GetRiskColour(
            string severity
        )
        {
            string normalised =
                severity?.Trim().ToLowerInvariant()
                ?? "";

            switch (normalised)
            {
                case "low":
                    return new Color(
                        1f,
                        0.82f,
                        0.1f,
                        1f
                    );

                case "medium":
                    return new Color(
                        1f,
                        0.48f,
                        0.05f,
                        1f
                    );

                case "high":
                    return new Color(
                        0.95f,
                        0.12f,
                        0.08f,
                        1f
                    );

                case "critical":
                    return new Color(
                        0.72f,
                        0.02f,
                        0.32f,
                        1f
                    );

                default:
                    return Color.white;
            }
        }
    }
}