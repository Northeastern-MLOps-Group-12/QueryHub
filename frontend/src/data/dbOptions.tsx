// Provider options for cloud services
export const providerOptions = ["gcp", "aws", "azure"];

// Database type options for each cloud provider
export const dbTypeOptions: Record<string, string[]> = {
  gcp: ["postgres", "MySQL", "SQL Server"],
  aws: [
    "MySQL",
    "postgres",
    "MariaDB",
    "Microsoft SQL Server",
    "Oracle",
    "Amazon Aurora",
  ],
  azure: [
    "Azure SQL Database",
    "Azure SQL Managed Instance",
    "SQL Server on Azure Virtual Machines",
    "Azure Database for PostgreSQL",
    "Azure Database for MySQL",
    "Azure Database for MariaDB",
  ],
};
