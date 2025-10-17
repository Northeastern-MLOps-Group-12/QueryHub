export const providerOptions = ["GCP", "AWS", "Azure"];

export const dbTypeOptions: Record<string, string[]> = {
  GCP: ["PostgreSQL", "MySQL", "SQL Server"],
  AWS: [
    "MySQL",
    "PostgreSQL",
    "MariaDB",
    "Microsoft SQL Server",
    "Oracle",
    "Amazon Aurora",
  ],
  Azure: [
    "Azure SQL Database",
    "Azure SQL Managed Instance",
    "SQL Server on Azure Virtual Machines",
    "Azure Database for PostgreSQL",
    "Azure Database for MySQL",
    "Azure Database for MariaDB",
  ],
};
