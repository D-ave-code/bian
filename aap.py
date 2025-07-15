import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def main():
    # Tus textos (modifica con tus datos)
    texts = [
    # Payments Execution
    "initiate payment transfer",
    "process payment confirmation",
    "validate payment method",
    "cancel payment transaction",
    "log payment gateway response",
    "execute funds transfer",
    "process wire transfer",
    "handle payment settlement",
    "process bulk payment files",
    "audit payment transactions",

    # Loan Contract Management
    "create new loan contract",
    "update loan agreement status",
    "calculate loan interest",
    "approve loan application",
    "fetch loan contract details",
    "process loan repayment",
    "evaluate credit risk for loan",
    "close loan account",
    "generate loan amortization schedule",
    "validate loan eligibility",

    # KYC & Compliance
    "validate customer identity documents",
    "process kyc verification",
    "send kyc document reminders",
    "check customer compliance status",
    "handle fraud detection alerts",
    "audit customer risk profile",
    "monitor transaction for suspicious activity",
    "perform customer due diligence",
    "register compliance exceptions",
    "report suspicious transactions",

    # Customer Relationship Management
    "register new customer",
    "update customer contact info",
    "track customer complaint status",
    "manage customer preferences",
    "send marketing offers",
    "handle customer loyalty program",
    "generate customer activity report",
    "log customer interactions",
    "manage customer account access",
    "notify customer about promotions",

    # Account Management
    "open new savings account",
    "close checking account",
    "reconcile account transactions",
    "generate monthly account statement",
    "handle account freeze requests",
    "calculate account fees",
    "update account holder information",
    "process term deposit maturity",
    "manage fixed deposit account",
    "schedule account reconciliation"
]



    # Vectorización TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # Clustering KMeans
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X)

    # Reducción de dimensionalidad con PCA
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X.toarray())

    # Graficar
    plt.figure(figsize=(8,6))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    for i in range(k):
        plt.scatter(
            X_reduced[labels == i, 0], 
            X_reduced[labels == i, 1], 
            label=f'Cluster {i}',
            color=colors[i]
        )

    for i, txt in enumerate(texts):
        plt.annotate(txt, (X_reduced[i, 0], X_reduced[i, 1]), fontsize=8)
    # Mostrar los clusters y sus elementos
    print(f"\n📋 Resultados del clustering con k={5}:")
    for i in range(5):
        cluster_texts = [texts[j] for j in range(len(texts)) if labels[j] == i]
        print(f"\n🔹 Cluster {i} ({len(cluster_texts)} elementos):")
        for text in cluster_texts:
            print(f"   • {text}")
    plt.title("K-Means Clustering de Textos (PCA 2D)")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
