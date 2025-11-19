import logging


def send_email_notification(subject, html_content, to_emails=None):
    """
    Utility function to send email notifications
    
    Args:
        subject (str): Email subject line
        html_content (str): HTML content for email body
        to_emails (list): List of recipient email addresses
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    from airflow.utils.email import send_email
    
    if to_emails is None:
        to_emails = ['jayjajoousa02@gmail.com']  # Default recipient
    
    try:
        send_email(
            to=to_emails,
            subject=subject,
            html_content=html_content
        )
        logging.info(f"✅ Email sent successfully to {', '.join(to_emails)}")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to send email: {e}")
        logging.info("💡 Tip: Configure SMTP settings in docker-compose.yaml or airflow.cfg")
        logging.info("💡 Example SMTP config for Gmail:")
        logging.info("   AIRFLOW__SMTP__SMTP_HOST=smtp.gmail.com")
        logging.info("   AIRFLOW__SMTP__SMTP_PORT=587")
        logging.info("   AIRFLOW__SMTP__SMTP_USER=your-email@gmail.com")
        logging.info("   AIRFLOW__SMTP__SMTP_PASSWORD=your-app-password")
        logging.info("   AIRFLOW__SMTP__SMTP_MAIL_FROM=your-email@gmail.com")
        return False


def notify_task_failure(context, to_emails=None):
    """
    Callback function to send email notification on task failure
    
    This function is designed to be used as an on_failure_callback
    in Airflow DAG default_args.
    
    Args:
        context (dict): Airflow context dictionary containing task instance info
    """
    task_instance = context.get('task_instance')
    dag_id = context.get('dag').dag_id
    task_id = task_instance.task_id
    execution_date = context.get('execution_date')
    exception = context.get('exception')
    log_url = task_instance.log_url
    
    # Get task duration if available
    try:
        duration = task_instance.duration
        duration_str = f"{duration:.2f} seconds" if duration else "N/A"
    except:
        duration_str = "N/A"
    
    subject = f"🚨 Airflow Task Failure: {dag_id}.{task_id}"
    
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .header {{ background-color: #e74c3c; color: white; padding: 20px; }}
            .content {{ padding: 20px; }}
            .error-box {{ background-color: #fee; border-left: 4px solid #e74c3c; padding: 15px; margin: 15px 0; }}
            .info-box {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #e74c3c; color: white; }}
            a {{ color: #3498db; text-decoration: none; }}
            code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>🚨 Airflow Task Failure Alert</h2>
            <p>A task in your QueryHub data pipeline has failed</p>
        </div>
        
        <div class="content">
            <div class="info-box">
                <h3>📋 Task Details</h3>
                <table>
                    <tr>
                        <th>Property</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td><strong>DAG ID</strong></td>
                        <td><code>{dag_id}</code></td>
                    </tr>
                    <tr>
                        <td><strong>Task ID</strong></td>
                        <td><code>{task_id}</code></td>
                    </tr>
                    <tr>
                        <td><strong>Execution Date</strong></td>
                        <td>{execution_date}</td>
                    </tr>
                    <tr>
                        <td><strong>Duration</strong></td>
                        <td>{duration_str}</td>
                    </tr>
                    <tr>
                        <td><strong>State</strong></td>
                        <td style="color: #e74c3c; font-weight: bold;">FAILED ❌</td>
                    </tr>
                </table>
            </div>
            
            <div class="error-box">
                <h3>❌ Error Details</h3>
                <pre style="white-space: pre-wrap; word-wrap: break-word; font-family: monospace; font-size: 13px;">{str(exception)}</pre>
            </div>
            
            <div class="info-box">
                <h3>🔍 Troubleshooting Steps</h3>
                <ol>
                    <li>Review the <a href="{log_url}" style="font-weight: bold;">detailed task logs</a> for complete error information</li>
                    <li>Check if upstream tasks completed successfully</li>
                    <li>Verify data quality and file availability</li>
                    <li>Check resource availability (memory, disk space, connections)</li>
                    <li>Review recent code changes that might affect this task</li>
                    <li>If the error is transient, try clearing and rerunning the task</li>
                </ol>
            </div>
            
            <div class="info-box">
                <h3>🔧 Quick Actions</h3>
                <ul>
                    <li><strong>View Logs:</strong> <a href="{log_url}">Click here to view task logs</a></li>
                    <li><strong>Retry Task:</strong> Clear the task state in Airflow UI and rerun</li>
                    <li><strong>Check Dependencies:</strong> Verify all upstream tasks succeeded</li>
                </ul>
            </div>
            
            <p style="margin-top: 30px; color: #7f8c8d; font-size: 12px;">
                <em>This is an automated alert from the QueryHub Data Pipeline</em><br>
                <em>To disable these notifications, remove the on_failure_callback from DAG default_args</em>
            </p>
        </div>
    </body>
    </html>
    """
    
    send_email_notification(subject, html_content, to_emails)
    logging.error(f"📧 Task failure notification sent for {dag_id}.{task_id}")


def generate_bias_detection_email(complexity_counts, total_samples, bias_info, minority_classes):
    """
    Generate HTML content for bias detection email
    
    Args:
        complexity_counts: pandas Series with complexity value counts
        total_samples (int): Total number of samples
        bias_info (dict): Dictionary with bias detection results
        minority_classes (list): List of minority class descriptions
        
    Returns:
        tuple: (subject, html_content)
    """
    bias_level = bias_info['bias_level']
    imbalance_ratio = bias_info['imbalance_ratio']
    max_class = bias_info['max_class']
    min_class = bias_info['min_class']
    max_count = bias_info['max_count']
    min_count = bias_info['min_count']
    
    percentages = (complexity_counts / total_samples * 100).round(2)
    
    subject = f"⚠️ Dataset Bias Detection Alert - {bias_level} Imbalance Detected"
    
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .header {{ background-color: #ff6b6b; color: white; padding: 20px; }}
            .content {{ padding: 20px; }}
            .stats {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .warning {{ color: #d63031; font-weight: bold; }}
            .info {{ color: #0984e3; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            .minority {{ background-color: #fff3cd; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>🔍 Dataset Bias Detection Report</h2>
            <p>SQL Complexity Class Imbalance Analysis</p>
        </div>
        
        <div class="content">
            <h3 class="warning">Bias Level: {bias_level}</h3>
            <p>Imbalance Ratio: <strong>{imbalance_ratio:.2f}x</strong></p>
            
            <div class="stats">
                <h4>📊 Dataset Statistics</h4>
                <ul>
                    <li>Total Training Samples: <strong>{total_samples:,}</strong></li>
                    <li>Majority Class: <strong>{max_class}</strong> ({max_count:,} samples, {percentages[max_class]}%)</li>
                    <li>Minority Class: <strong>{min_class}</strong> ({min_count:,} samples, {percentages[min_class]}%)</li>
                    <li>Number of Complexity Types: <strong>{len(complexity_counts)}</strong></li>
                </ul>
            </div>
            
            <h4>📈 Complete Distribution</h4>
            <table>
                <tr>
                    <th>SQL Complexity</th>
                    <th>Sample Count</th>
                    <th>Percentage</th>
                    <th>Status</th>
                </tr>
    """
    
    MINORITY_THRESHOLD = 0.5
    for complexity in complexity_counts.index:
        count = complexity_counts[complexity]
        pct = percentages[complexity]
        is_minority = count < (max_count * MINORITY_THRESHOLD)
        row_class = "minority" if is_minority else ""
        status = "⚠️ Minority" if is_minority else "✅ Adequate"
        
        html_content += f"""
                <tr class="{row_class}">
                    <td>{complexity}</td>
                    <td>{count:,}</td>
                    <td>{pct}%</td>
                    <td>{status}</td>
                </tr>
        """
    
    html_content += f"""
            </table>
            
            <div class="stats">
                <h4>⚠️ Identified Issues</h4>
                <ul>
                    <li><strong>{len(minority_classes)}</strong> minority class(es) detected (< 50% of majority class)</li>
                    {'<li class="warning">Classes: ' + ', '.join(minority_classes) + '</li>' if minority_classes else '<li class="info">No severe minority classes found</li>'}
                </ul>
            </div>
            
            <div class="stats">
                <h4>🔧 Recommended Actions</h4>
                <ul>
    """
    
    if bias_level == "SEVERE":
        html_content += """
                    <li>🚨 <strong>URGENT:</strong> Severe class imbalance detected</li>
                    <li>Generate synthetic data to balance minority classes</li>
                    <li>Consider oversampling techniques (SMOTE, ADASYN)</li>
                    <li>Use class weights in model training</li>
                    <li>Evaluate model performance per class, not just overall accuracy</li>
        """
    elif bias_level == "MODERATE":
        html_content += """
                    <li>⚠️ Moderate imbalance - synthetic data generation recommended</li>
                    <li>Monitor model performance on minority classes</li>
                    <li>Consider stratified sampling during training</li>
                    <li>Apply class weights if model supports it</li>
        """
    else:
        html_content += """
                    <li>✅ Mild imbalance detected - proceeding with synthetic data generation</li>
                    <li>Continue monitoring class distribution in future runs</li>
                    <li>Validate model performance across all complexity types</li>
        """
    
    html_content += """
                </ul>
            </div>
            
            <p style="margin-top: 20px;"><strong>Next Step:</strong> Pipeline will automatically generate synthetic data to balance the dataset.</p>
            
            <p style="margin-top: 30px; color: #7f8c8d; font-size: 12px;">
                <em>This is an automated alert from the QueryHub Data Pipeline</em>
            </p>
        </div>
    </body>
    </html>
    """
    
    return subject, html_content