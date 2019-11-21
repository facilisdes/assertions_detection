<VirtualHost *:80>
       ServerName localhost
       ServerAlias localhost
       DocumentRoot /var/www/project/project/public

       <Directory /var/www/project/project>
            Options Indexes FollowSymLinks
            AllowOverride None
            Require all granted
        </Directory>
</VirtualHost>
